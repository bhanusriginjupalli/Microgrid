"""
experiments/phase4/robustness/run_training_experiments.py
============================================================
Trains multi-seed Robust MAPPO policies and ablation models:
  - Seed 101 (Independent replication 1)
  - Seed 202 (Independent replication 2)
  - Ablation 1: no_sensor_noise (Domain Randomization without sensor noise)
  - Ablation 2: no_uncertainty (Forecast-Aware MAPPO without uncertainty intervals)
============================================================
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.phase4.robustness.train_robust_mappo import train_robust_mappo

def run_all():
    print("=================================================================")
    print("GRIDGUARD AI - Training Multi-Seed Robust MAPPO & Ablations")
    print("=================================================================")
    
    # 1. Independent Seed 101
    print("\n>>> Launching Seed 101 Training...")
    train_robust_mappo(seed=101, episodes=40, ablation="none")
    
    # 2. Independent Seed 202
    print("\n>>> Launching Seed 202 Training...")
    train_robust_mappo(seed=202, episodes=40, ablation="none")

    # 3. Ablation: No Sensor Noise
    print("\n>>> Launching Ablation: No Sensor Noise...")
    train_robust_mappo(seed=42, episodes=35, ablation="no_sensor_noise")

    # 4. Ablation: No Uncertainty
    print("\n>>> Launching Ablation: No Uncertainty...")
    train_robust_mappo(seed=42, episodes=35, ablation="no_uncertainty")

    print("\n=================================================================")
    print("All Multi-Seed and Ablation Policies Successfully Trained!")
    print("=================================================================")

if __name__ == "__main__":
    run_all()
