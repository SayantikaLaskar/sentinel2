"""
SENTINEL — GRPO Training Launcher
---------------------------------
This script initializes the Llama-3.2-1B model and starts the 
Reasoning-Augmented GRPO training loop.
"""
import os
import sys
import subprocess

# Force UTF-8 mode on Windows to prevent 'charmap' codec errors in TRL
if os.name == 'nt' and os.environ.get("PYTHONUTF8") != "1":
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    sys.exit(subprocess.call([sys.executable] + sys.argv, env=env))

import logging
from sentinel.training.pipeline import run_training_loop, TrainingConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SENTINEL-Training")

def main():
    print("=" * 60)
    print("  SENTINEL: LLM + MATH HYBRID TRAINING")
    print("=" * 60)
    
    # 1. Setup Configuration for RTX 3050 (4GB VRAM)
    config = TrainingConfig(
        agent="holmes",            # Default agent
        model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        max_steps=100000,          # Continuous hackathon mode
        batch_size=1,              # Smallest footprint for 4GB VRAM
        gradient_accumulation_steps=4,
    )
    
    print(f"  Target Model: {config.model_name}")
    print(f"  Max Steps:    {config.max_steps}")
    print(f"  Batch Size:   {config.batch_size}")
    print("-" * 60)
    
    # 2. Instantiate Environment and Reward Function
    from sentinel.env import Sentinel_Env
    
    print("  Initializing environment...")
    env = Sentinel_Env(render_mode="human")
    reward_fn = env.reward_function
    
    # 3. Build Trainer (Hybrid Math/LLM) — only if GPU is available
    from sentinel.training.pipeline import build_grpo_trainer
    import torch
    
    if torch.cuda.is_available():
        print("  Building Hybrid Trainer (GPU detected)...")
        trainer, llm_agent = build_grpo_trainer(agent=config.agent, env=env, config=config)
    else:
        print("  No GPU detected — running in fast UCB1+Bayesian math mode (skipping LLM load).")
        trainer, llm_agent = None, None
    
    # 4. Start Training
    try:
        print("  Starting GRPO loop...")
        run_training_loop(
            env=env,
            trainer=trainer,
            llm_agent=llm_agent,
            config=config,
            reward_fn=reward_fn
        )
    except Exception as e:
        print(f"\n[!] Training Error: {e}")
        print("\nPossible solutions:")
        print("1. Ensure 'pip install trl unsloth' completed.")
        print("2. Check if another app is using your GPU VRAM.")
        print("3. Try decreasing max_seq_length in TrainingConfig.")

if __name__ == "__main__":
    main()
