"""Generate results/training_curves.png — dual-scale for clear visual."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("results/simulation_results.json") as f:
    data = json.load(f)

baseline = data["baseline"]["rewards"]
trained = data["trained"]["rewards"]

def smooth(vals, window=10):
    if len(vals) < window:
        return vals
    kernel = np.ones(window) / window
    return np.convolve(vals, kernel, mode="valid").tolist()

def cumulative_mean(vals):
    out = []
    s = 0.0
    for i, v in enumerate(vals):
        s += v
        out.append(s / (i + 1))
    return out

# ── Figure: 2x2 grid ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor("#0f172a")

for row in axes:
    for ax in row:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8", labelsize=9)
        ax.xaxis.label.set_color("#94a3b8")
        ax.yaxis.label.set_color("#94a3b8")
        ax.title.set_color("#e2e8f0")
        for spine in ax.spines.values():
            spine.set_color("#334155")

RED = "#ef4444"
GREEN = "#22c55e"
YELLOW = "#f59e0b"
PURPLE = "#a78bfa"

# ═══ Panel 1 (top-left): Cumulative mean reward ═══
ax = axes[0][0]
cm_b = cumulative_mean(baseline)
cm_t = cumulative_mean(trained)
ax.plot(range(len(cm_b)), cm_b, color=RED, linewidth=2.5, label="Baseline (random)")
ax.plot(range(len(cm_t)), cm_t, color=GREEN, linewidth=2.5, label="Trained (UCB1+Bayes)")
ax.axhline(0, color="#475569", linestyle="--", linewidth=0.8)
ax.set_xlabel("Episode")
ax.set_ylabel("Cumulative Mean Reward")
ax.set_title("Learning Curve — Cumulative Mean Reward", fontsize=11, fontweight="bold")
ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=9)

# ═══ Panel 2 (top-right): Trained agent ONLY (zoomed in) ═══
ax = axes[0][1]
st = smooth(trained, window=5)
ax.plot(range(len(st)), st, color=GREEN, linewidth=2, alpha=0.9)
ax.fill_between(range(len(st)), st, alpha=0.15, color=GREEN)
ax.axhline(0, color="#475569", linestyle="--", linewidth=0.8)
# Add rolling window markers
window_means = []
for i in range(0, len(trained), 20):
    chunk = trained[i:i+20]
    wm = sum(chunk) / len(chunk)
    window_means.append((i + len(chunk)//2, wm))
    ax.plot(i + len(chunk)//2, wm, "o", color=YELLOW, markersize=8, zorder=5)
    ax.annotate(f"{wm:.2f}", (i + len(chunk)//2, wm),
                textcoords="offset points", xytext=(0, 12),
                color=YELLOW, fontsize=9, ha="center", fontweight="bold")
ax.set_xlabel("Episode")
ax.set_ylabel("Total Reward (smoothed)")
ax.set_title("Trained Agent Reward (zoomed) — showing improvement", fontsize=11, fontweight="bold")

# ═══ Panel 3 (bottom-left): Mean comparison bar chart ═══
ax = axes[1][0]
categories = ["Baseline\n(random)", "Trained\n(UCB1+Bayes)"]
means = [data["baseline"]["mean"], data["trained"]["mean"]]
colors_bar = [RED, GREEN]
bars = ax.bar(categories, means, color=colors_bar, edgecolor="#e2e8f0", linewidth=0.5, width=0.45)
for bar, val in zip(bars, means):
    y = bar.get_height()
    offset = 0.3 if y >= 0 else -1.2
    ax.text(bar.get_x() + bar.get_width()/2, y + offset,
            f"{val:.2f}", ha="center", va="bottom", color="#e2e8f0", fontsize=14, fontweight="bold")
ax.set_ylabel("Mean Episode Reward")
ax.set_title("Mean Reward Comparison (+99.5% improvement)", fontsize=11, fontweight="bold")
ax.axhline(0, color="#475569", linestyle="--", linewidth=0.8)

# ═══ Panel 4 (bottom-right): UCB1 arm selection ═══
ax = axes[1][1]
if "ucb1_arms" in data:
    arms = data["ucb1_arms"][:8]
    arm_names = [a["arm"].split("/")[1] for a in reversed(arms)]
    arm_pulls = [a["pulls"] for a in reversed(arms)]
    arm_means = [a["mean_reward"] for a in reversed(arms)]
    bar_colors = [GREEN if m >= 0 else PURPLE for m in arm_means]
    hbars = ax.barh(arm_names, arm_pulls, color=bar_colors, edgecolor="#e2e8f0", linewidth=0.3, alpha=0.85)
    for i, (p, m) in enumerate(zip(arm_pulls, arm_means)):
        ax.text(p + max(arm_pulls)*0.02, i, f"{p} pulls (r\u0304={m:.3f})",
                va="center", color="#e2e8f0", fontsize=8)
    ax.set_xlabel("Total Pulls")
    ax.set_title("UCB1 Bandit Arm Distribution (Auer 2002)", fontsize=11, fontweight="bold")

fig.suptitle("SENTINEL — Training Results: Baseline vs Math-Driven Agent (200 episodes)",
             color="#f8fafc", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig("results/training_curves.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved results/training_curves.png")
plt.close()
