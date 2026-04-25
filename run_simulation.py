"""Run 200 episodes: 100 baseline (random) + 100 trained (UCB1+Bayesian RCA).
Saves results/simulation_results.json with both runs for plotting."""
import json
import os
from sentinel.env import Sentinel_Env
from sentinel.training.pipeline import _get_action, TrainingConfig
from sentinel.math_engine import get_ucb1_bandit, get_bayesian_rca

os.makedirs("results", exist_ok=True)

N_EPISODES = 100

# ═══════════════════════════════════════════════════════════════════
# PHASE 1: BASELINE — Random cycling actions (no intelligence)
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("  PHASE 1: BASELINE (random actions) — 100 episodes")
print("=" * 60)

RANDOM_ACTIONS = [
    {"agent": "holmes", "category": "investigative", "name": "QueryLogs",
     "params": {"service": "api-gateway", "time_range": [0, 60]}},
    {"agent": "holmes", "category": "investigative", "name": "QueryMetrics",
     "params": {"service": "web-gateway", "metric_name": "cpu", "time_range": [0, 300]}},
    {"agent": "forge", "category": "remediation", "name": "RestartService",
     "params": {"service": "api-gateway"}},
    {"agent": "forge", "category": "remediation", "name": "ScaleService",
     "params": {"service": "web-gateway", "replicas": 2}},
]

baseline_env = Sentinel_Env(config_path="env_spec.yaml", incident_library_path="incident_library.yaml")
baseline_rewards = []

for ep in range(N_EPISODES):
    obs, info = baseline_env.reset(seed=ep)
    total_r = 0.0
    for step in range(60):
        action = RANDOM_ACTIONS[step % len(RANDOM_ACTIONS)]
        obs, r, term, trunc, _ = baseline_env.step(action)
        total_r += float(r)
        if term or trunc:
            break
    baseline_rewards.append(round(total_r, 4))
    if (ep + 1) % 20 == 0:
        avg = sum(baseline_rewards[-20:]) / 20
        print(f"  Episode {ep+1:3d} | avg(last 20) = {avg:.4f}")

b_mean = sum(baseline_rewards) / len(baseline_rewards)
print(f"\n  Baseline mean reward: {b_mean:.4f}")

# ═══════════════════════════════════════════════════════════════════
# PHASE 2: TRAINED — UCB1 bandit + Bayesian RCA (math engine)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PHASE 2: TRAINED (UCB1 + Bayesian RCA) — 100 episodes")
print("=" * 60)

trained_env = Sentinel_Env(config_path="env_spec.yaml", incident_library_path="incident_library.yaml")
cfg = TrainingConfig(agent="holmes")
bandit = get_ucb1_bandit()
trained_rewards = []

for ep in range(N_EPISODES):
    obs, info = trained_env.reset(seed=100 + ep)
    total_r = 0.0
    for step in range(60):
        action = _get_action(None, obs, cfg)
        arm_idx = action.pop("_ucb1_arm_idx", None)
        obs, r, term, trunc, _ = trained_env.step(action)
        if arm_idx is not None:
            bandit.update(arm_idx, float(r))
        total_r += float(r)
        if term or trunc:
            break
    trained_rewards.append(round(total_r, 4))
    if (ep + 1) % 20 == 0:
        avg = sum(trained_rewards[-20:]) / 20
        print(f"  Episode {ep+1:3d} | avg(last 20) = {avg:.4f}")

t_mean = sum(trained_rewards) / len(trained_rewards)
print(f"\n  Trained mean reward: {t_mean:.4f}")

# ═══════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════
results = {
    "n_episodes": N_EPISODES,
    "baseline": {
        "rewards": baseline_rewards,
        "mean": round(b_mean, 4),
        "min": round(min(baseline_rewards), 4),
        "max": round(max(baseline_rewards), 4),
    },
    "trained": {
        "rewards": trained_rewards,
        "mean": round(t_mean, 4),
        "min": round(min(trained_rewards), 4),
        "max": round(max(trained_rewards), 4),
    },
    "ucb1_arms": bandit.arm_stats(),
}

with open("results/simulation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("  RESULTS SAVED")
print("=" * 60)
print(f"  Baseline mean: {b_mean:.4f}")
print(f"  Trained mean:  {t_mean:.4f}")
if b_mean != 0:
    pct = ((t_mean - b_mean) / abs(b_mean)) * 100
    print(f"  Improvement:   {pct:+.1f}%")
print(f"\n  File: results/simulation_results.json")
print(f"  Next: python plot_results.py")
