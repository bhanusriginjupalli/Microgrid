"""
controller_registry.py
======================
Unified controller loader and wrapper for all 6 Phase 3 study controllers:
  1. IPPO (IPPOAgent + IPPOCheckpointManager)
  2. MAPPO_MATCHED (MAPPOAgent via load_mappo_checkpoint)
  3. HAPPO (HAPPOAgent via load_happo_checkpoint)
  4. Greedy EMS (GreedyController)
  5. Rule-Based EMS (RuleBasedEMS)
  6. Lookahead Heuristic EMS (MPCController(horizon_steps=8))
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "phase1_digital_twin") not in sys.path:
    sys.path.insert(0, str(ROOT / "phase1_digital_twin"))

from rl_environment.microgrid_parallel_env import MicrogridParallelEnv
from controllers.greedy_controller import GreedyController
from controllers.rule_based_ems import RuleBasedEMS
from controllers.mpc_controller import MPCController
from ippo import IPPOConfig, IPPOAgent, IPPOCheckpointManager
from happo import load_happo_checkpoint
from experiments.phase4_4_3.benchmark_phase4_4_3 import load_mappo_checkpoint


def load_ippo_checkpoint(
    checkpoint_path: str | Path,
    forecast_mode: Optional[str] = "realistic",
    device: str = "cpu",
) -> IPPOAgent:
    """Load an independently saved IPPO checkpoint for deterministic inference."""
    ckpt_path = Path(checkpoint_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = dict(ckpt.get("config", {}))
    valid = set(IPPOConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in cfg_dict.items() if k in valid}
    filtered["forecast_mode"] = forecast_mode
    filtered["device"] = device
    cfg = IPPOConfig(**filtered)

    models_dir = cfg.forecast_models_dir or str(ROOT / "experiments" / "phase4" / "forecasting" / "models")
    dummy_env = MicrogridParallelEnv(
        scenario_id="scenario_a",
        forecast_mode=forecast_mode,
        forecast_models_dir=models_dir,
    )
    obs_dims = {ag: int(dummy_env.observation_space(ag).shape[0]) for ag in dummy_env.possible_agents}
    agent = IPPOAgent(config=cfg, obs_dims=obs_dims, action_spaces=dummy_env._action_spaces)
    IPPOCheckpointManager(ckpt_path.parent).load_checkpoint(ckpt_path, agent, load_optimizers=False)
    agent.eval()
    return agent


@dataclass
class ControllerSpec:
    controller_name: str
    algorithm: str
    algorithm_slug: str
    controller_group: str  # "Learned MARL" | "Heuristic Baseline"
    train_seed: str        # "42" | "101" | "202" | "deterministic"
    is_learned: bool
    checkpoint_path: Optional[str]
    obj: Any

    def select_actions(
        self,
        obs: Dict[str, np.ndarray],
        prev_state: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float]:
        """Execute deterministic inference and return (actions, inference_latency_ms)."""
        t0 = time.perf_counter()
        if self.is_learned:
            actions, _ = self.obj.select_actions(obs, deterministic=True)
        else:
            actions = self.obj.select_actions(obs, prev_state)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return actions, latency_ms


def build_controller_specs(
    results_root: Path,
    checkpoint_tag: str = "best_model.pt",
    train_seeds: Optional[List[int]] = None,
    include_heuristics: bool = True,
    include_learned: bool = True,
    use_smoke_if_missing: bool = False,
) -> List[ControllerSpec]:
    """Build exact controller roster from trained checkpoints + deterministic heuristics."""
    seeds = list(train_seeds or [42, 101, 202])
    specs: List[ControllerSpec] = []

    if include_learned:
        algo_loaders = [
            ("IPPO", "ippo", "ippo", load_ippo_checkpoint),
            ("MAPPO_MATCHED", "mappo_matched", "mappo_matched", load_mappo_checkpoint),
            ("HAPPO", "happo", "happo", load_happo_checkpoint),
        ]
        for algo_name, slug, subdir, loader_fn in algo_loaders:
            for s in seeds:
                ckpt = results_root / subdir / f"seed_{s}" / "checkpoints" / checkpoint_tag
                if not ckpt.exists() and use_smoke_if_missing:
                    ckpt = results_root / "smoke" / subdir / f"seed_{s}" / "checkpoints" / checkpoint_tag
                if not ckpt.exists():
                    raise FileNotFoundError(
                        f"Missing checkpoint for {algo_name} seed={s}: {ckpt}"
                    )
                obj = loader_fn(ckpt, forecast_mode="realistic")
                specs.append(
                    ControllerSpec(
                        controller_name=f"{algo_name} (Seed {s})",
                        algorithm=algo_name,
                        algorithm_slug=slug,
                        controller_group="Learned MARL",
                        train_seed=str(s),
                        is_learned=True,
                        checkpoint_path=str(ckpt),
                        obj=obj,
                    )
                )

    if include_heuristics:
        specs.extend([
            ControllerSpec(
                controller_name="Greedy EMS",
                algorithm="Greedy EMS",
                algorithm_slug="greedy_ems",
                controller_group="Heuristic Baseline",
                train_seed="deterministic",
                is_learned=False,
                checkpoint_path=None,
                obj=GreedyController(name="Greedy EMS"),
            ),
            ControllerSpec(
                controller_name="Rule-Based EMS",
                algorithm="Rule-Based EMS",
                algorithm_slug="rule_based_ems",
                controller_group="Heuristic Baseline",
                train_seed="deterministic",
                is_learned=False,
                checkpoint_path=None,
                obj=RuleBasedEMS(name="Rule-Based EMS"),
            ),
            ControllerSpec(
                controller_name="Lookahead Heuristic EMS",
                algorithm="Lookahead Heuristic EMS",
                algorithm_slug="lookahead_heuristic_ems",
                controller_group="Heuristic Baseline",
                train_seed="deterministic",
                is_learned=False,
                checkpoint_path=None,
                obj=MPCController(horizon_steps=8, name="Lookahead Heuristic EMS"),
            ),
        ])

    return specs
