"""SENTINEL Gradio Dashboard.

Visualizes NexaStack health, agent actions, and training progress.
Compatible with HuggingFace Spaces (demo.launch() at module level).
"""
from __future__ import annotations

import json
import time
from typing import Any

import sys
import os
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "numpy"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')
import networkx as nx
from matplotlib.figure import Figure
from PIL import Image
import io

from sentinel.config import load_config
from sentinel.training.pipeline import _get_action, TrainingConfig

try:
    import gradio as gr
    _GRADIO_AVAILABLE = True
except ImportError:
    gr = None  # type: ignore[assignment]
    _GRADIO_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_action_log: list[dict] = []          # last 20 actions
_metrics_log: list[dict] = []         # last 50 episode metrics
_oracle_gap: str = "No episodes completed yet."
_env: Any = None  # Sentinel_Env | None

# Incident IDs from the library
_INCIDENT_IDS = ["E1", "E2", "E3", "M1", "M2", "M3", "M4", "H1", "H2", "H3"]

# ---------------------------------------------------------------------------
# Helper: seed demo state
# ---------------------------------------------------------------------------

def _seed_demo_state(env: Any) -> None:
    """Run 5 steps with seed=42 to populate demo state using smart actions."""
    cfg = TrainingConfig(agent="holmes")
    obs, info = env.reset(seed=42)
    for _ in range(5):
        action = _get_action(None, obs, cfg)
        arm_idx = action.pop("_ucb1_arm_idx", None)
        obs, reward, terminated, truncated, _ = env.step(action)
        _action_log.append({
            "timestamp": time.time(),
            "agent": action.get("agent", "holmes"),
            "name": action.get("name", "QueryLogs"),
            "params": action.get("params", {}),
        })
        if terminated:
            break


def _create_demo_env() -> Any:
    """Create and seed a demo Sentinel_Env, returning None on failure."""
    try:
        from sentinel.env import Sentinel_Env
        env = Sentinel_Env()
        _seed_demo_state(env)
        templates = getattr(env.incident_generator, "_templates", [])
        if templates:
            global _INCIDENT_IDS
            _INCIDENT_IDS = [tpl.id for tpl in templates]
        return env
    except Exception as e:
        print(f"Warning: Could not create demo env: {e}")
        return None


# ---------------------------------------------------------------------------
# Dashboard component builders
# ---------------------------------------------------------------------------

def _tail_training_log() -> str:
    log_path = "training_log.jsonl"
    if not os.path.exists(log_path):
        return "No thoughts logged yet."
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
        parts = content.split("--- THOUGHT ---")
        if len(parts) > 1:
            last_thought = parts[-1].split("---------------")[0].strip()
            return last_thought
    except Exception:
        pass
    return "No thoughts logged yet."


def _tail_live_actions(n: int = 20) -> str:
    """Read the last n episode summaries from training_log.jsonl as a live feed."""
    log_path = "training_log.jsonl"
    if not os.path.exists(log_path):
        return build_action_feed(_action_log)
    try:
        records = []
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    rec = json.loads(line)
                    if "episode" in rec and "total_reward" in rec:
                        records.append(rec)
                except json.JSONDecodeError:
                    continue
        if not records:
            return build_action_feed(_action_log)
        recent = records[-n:][::-1]
        lines = []
        for rec in recent:
            ep = rec.get("episode", "?")
            r = rec.get("total_reward", 0.0)
            mttr = rec.get("mttr", "?")
            r3 = rec.get("r3", 0.0)
            lines.append(f"[Ep {ep:>5}] reward={r:.3f} | recovery={r3:.2f} | MTTR={mttr}")
        return "\n".join(lines)
    except Exception:
        return build_action_feed(_action_log)

def build_causal_graph(env: Any) -> Any:
    if env is None or getattr(env.world_state, "cdg", None) is None:
        fig = Figure(figsize=(6, 4))
        ax = fig.subplots()
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#111827')
        ax.text(0.5, 0.5, "No Causal Graph Available", ha="center", va="center", color="white")
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
        buf.seek(0)
        import numpy as np
        from PIL import Image as _PILImage
        img = _PILImage.open(buf)
        img.load()
        return np.array(img)
    
    cdg = env.world_state.cdg
    blast = env.world_state.incident_state.current_blast_radius if env.world_state.incident_state else set()
    
    colors = []
    for node in cdg.nodes():
        if node in blast:
            colors.append("#ef4444")
        elif env.world_state.services[node].availability is False:
            colors.append("#f97316")
        else:
            colors.append("#22c55e")
            
    fig = Figure(figsize=(8, 6))
    fig.patch.set_facecolor('#111827')
    ax = fig.subplots()
    ax.set_facecolor('#111827')
    
    pos = nx.spring_layout(cdg, seed=42)
    nx.draw(
        cdg, pos, ax=ax, node_color=colors,
        with_labels=True, node_size=800,
        font_size=8, font_color="white", font_weight="bold",
        edge_color="#4b5563", width=1.5, arrowsize=15
    )
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    import numpy as np
    from PIL import Image as _PILImage
    img = _PILImage.open(buf)
    img.load()
    return np.array(img)

def build_health_grid(env: Any) -> str:
    """Return an HTML string with a 30-service color-coded grid."""
    if env is None:
        return "<p style='color:gray'>No environment available.</p>"

    blast_radius: set[str] = set()
    if env.world_state.incident_state is not None:
        blast_radius = env.world_state.incident_state.current_blast_radius

    cells = []
    for svc, metrics in env.world_state.services.items():
        bg = "#22c55e" if metrics.availability else "#ef4444"
        border = "3px solid #f97316" if svc in blast_radius else "1px solid #374151"
        cpu_pct = round(metrics.cpu * 100, 1)
        err_pct = round(metrics.error_rate * 100, 2)
        cell = (
            f"<div style='"
            f"background:{bg};border:{border};border-radius:6px;"
            f"padding:6px 4px;margin:3px;display:inline-block;"
            f"width:140px;vertical-align:top;font-size:11px;color:#fff;"
            f"font-family:monospace;'>"
            f"<b>{svc}</b><br>"
            f"CPU {cpu_pct}% | Err {err_pct}%"
            f"</div>"
        )
        cells.append(cell)

    return (
        "<div style='background:#111827;padding:10px;border-radius:8px;"
        "line-height:1.6;'>"
        + "".join(cells)
        + "</div>"
    )


def build_action_feed(action_log: list[dict]) -> str:
    """Return last 20 entries from action_log as a formatted string (newest first)."""
    recent = action_log[-20:][::-1]
    if not recent:
        return "(no actions yet)"
    lines = []
    for entry in recent:
        ts = time.strftime("%H:%M:%S", time.localtime(entry.get("timestamp", 0)))
        agent = entry.get("agent", "?").upper()
        name = entry.get("name", "?")
        params = entry.get("params", {})
        # Summarise params: show first key=value pair only
        if params:
            first_key = next(iter(params))
            params_summary = f"{first_key}={params[first_key]}"
        else:
            params_summary = ""
        lines.append(f"[{ts}] {agent}: {name} ({params_summary})")
    return "\n".join(lines)


def _load_metrics_from_log(log_file: str = "training_log.jsonl") -> list[dict]:
    """Read episode metrics from training_log.jsonl written by run_training.py."""
    records: list[dict] = []
    if not os.path.exists(log_file):
        return records
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("---") or line.startswith("{") is False:
                    continue
                try:
                    rec = json.loads(line)
                    if "episode" in rec and "total_reward" in rec:
                        records.append(rec)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return records


def get_training_data(metrics_log: list[dict]) -> Any:
    """Return last 50 entries as a pandas DataFrame for LinePlot."""
    try:
        import pandas as pd
    except ImportError:
        return None

    # First try the live file written by run_training.py
    file_records = _load_metrics_from_log()
    recent = (file_records if file_records else metrics_log)[-50:]
    if not recent:
        return pd.DataFrame(columns=["episode", "total_reward", "mttr", "r1", "r2", "r3", "r4"])

    rows = []
    for i, m in enumerate(recent):
        rows.append({
            "episode": m.get("episode", i),
            "total_reward": m.get("total_reward", 0.0),
            "mttr": m.get("mttr", 0.0),
            "r1": m.get("r1", 0.0),
            "r2": m.get("r2", 0.0),
            "r3": m.get("r3", 0.0),
            "r4": m.get("r4", 0.0),
        })
    return pd.DataFrame(rows)


def build_oracle_display(oracle_gap: str) -> str:
    """Return formatted HTML for the ORACLE capability gap."""
    return (
        "<div style='background:#1e293b;border-radius:8px;padding:12px;"
        "font-family:monospace;color:#e2e8f0;font-size:12px;'>"
        "<b style='color:#a78bfa'>ORACLE Capability Gap</b><br><br>"
        f"{oracle_gap}"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Inject incident handler
# ---------------------------------------------------------------------------

def inject_incident(incident_id: str) -> str:
    """Inject a named incident into the environment."""
    global _env
    if _env is None:
        return "No environment available."
    try:
        # Filter the incident generator to only sample the chosen incident
        original_templates = _env.incident_generator._templates
        matching = [inc for inc in original_templates if inc.id == incident_id]
        if not matching:
            return f"Incident '{incident_id}' not found in library."

        _env.incident_generator._templates = matching
        original_dist = _env._difficulty_distribution
        _env._difficulty_distribution = {matching[0].difficulty: 1.0}
        cfg = load_config()
        _env.reset(seed=cfg.demo.seed)
        _env._difficulty_distribution = original_dist
        _env.incident_generator._templates = original_templates

        blast = _env.world_state.incident_state.current_blast_radius if _env.world_state.incident_state else set()
        return (
            f"Injected incident {incident_id}. "
            f"Blast radius: {len(blast)} services ({', '.join(sorted(blast)[:5])}{'...' if len(blast) > 5 else ''})"
        )
    except Exception as e:
        return f"Error injecting incident: {e}"


# ---------------------------------------------------------------------------
# Refresh callback
# ---------------------------------------------------------------------------

def _refresh() -> tuple[str, str, str, Any, str, Any]:
    """Return updated dashboard data for all auto-refresh components."""
    health_html = build_health_grid(_env)
    feed_text = _tail_live_actions()
    oracle_html = build_oracle_display(_oracle_gap)
    df = get_training_data(_metrics_log)
    cot_text = _tail_training_log()
    causal_fig = build_causal_graph(_env)
    return health_html, feed_text, oracle_html, df, cot_text, causal_fig


# ---------------------------------------------------------------------------
# build_dashboard
# ---------------------------------------------------------------------------

def build_dashboard(env: Any = None) -> Any:
    """Build and return the Gradio dashboard.

    If env is None, creates a pre-seeded Sentinel_Env for demo purposes.
    Returns None if Gradio is not installed.
    """
    if not _GRADIO_AVAILABLE:
        return None

    global _env
    if env is not None:
        _env = env
    elif _env is None:
        _env = _create_demo_env()

    with gr.Blocks(title="SENTINEL — Multi-Agent Incident Response") as dashboard:
        gr.Markdown("# SENTINEL — Multi-Agent Incident Response")

        # Row 1: Health grid
        with gr.Row():
            health_grid = gr.HTML(
                value=build_health_grid(_env),
                label="NexaStack Service Health",
            )

        # Row 2: Causal Graph | Action Feed | Agent CoT
        with gr.Row():
            with gr.Column(scale=2):
                causal_plot = gr.Image(
                    value=build_causal_graph(_env),
                    label="Causal Dependency Graph",
                    interactive=False,
                    type="numpy"
                )
            with gr.Column(scale=1):
                action_feed = gr.Textbox(
                    value=build_action_feed(_action_log),
                    label="Agent Action Feed",
                    lines=14,
                    interactive=False,
                )
            with gr.Column(scale=1):
                cot_display = gr.Textbox(
                    value=_tail_training_log(),
                    label="Live Agent Thoughts (CoT)",
                    lines=14,
                    interactive=False,
                )

        # Row 2.5: ORACLE gap
        with gr.Row():
            oracle_display = gr.HTML(
                value=build_oracle_display(_oracle_gap),
                label="ORACLE Capability Gap",
            )

        # Row 3: Training progress
        with gr.Row():
            try:
                import pandas as pd
                training_plot = gr.LinePlot(
                    value=get_training_data(_metrics_log),
                    x="episode",
                    y="total_reward",
                    label="Training Progress (Episode Reward)",
                )
            except Exception:
                training_plot = gr.Dataframe(
                    value=get_training_data(_metrics_log),
                    label="Training Metrics",
                )

        # Row 4: Inject incident
        with gr.Row():
            incident_dropdown = gr.Dropdown(
                choices=_INCIDENT_IDS,
                value=_INCIDENT_IDS[0],
                label="Incident ID",
            )
            inject_btn = gr.Button("Inject Incident")
            inject_status = gr.Textbox(
                value="",
                label="Status",
                interactive=False,
            )

        inject_btn.click(
            fn=inject_incident,
            inputs=[incident_dropdown],
            outputs=[inject_status],
        )

        # Auto-refresh every 2 seconds
        try:
            timer = gr.Timer(value=2.0)
            timer.tick(
                fn=_refresh,
                outputs=[health_grid, action_feed, oracle_display, training_plot, cot_display, causal_plot],
            )
        except Exception:
            # Fallback: use every= parameter if gr.Timer is unavailable
            health_grid.change(fn=_refresh, outputs=[health_grid, action_feed, oracle_display, training_plot, cot_display, causal_plot])

    return dashboard


# ---------------------------------------------------------------------------
# Module-level launch (required for HuggingFace Spaces auto-detection)
# ---------------------------------------------------------------------------

_env = _create_demo_env()
demo = build_dashboard(_env)

if demo is not None:
    demo.launch(share=False)
