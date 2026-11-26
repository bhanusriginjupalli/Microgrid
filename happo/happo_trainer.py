"""
happo/happo_trainer.py
======================
Matched training runner for Heterogeneous-Agent PPO (HAPPO) using the frozen
MicrogridParallelEnv, reward_v3_social.yaml, DomainRandomizer curriculum,
and FixedValidationSuite.
"""

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from rl_environment.microgrid_parallel_env import MicrogridParallelEnv
from rl_environment.economic_accounting import compute_episode_economics, EconomicConfig
from happo.happo_config import HAPPOConfig
from happo.happo_agent import HAPPOAgent
from happo.happo_buffer import HAPPORolloutBuffer
from happo.happo_advantage import (
    finalize_episode_happo_gae,
    normalize_happo_advantages,
)
from happo.happo_update import update_happo
from happo.happo_checkpoint import HAPPOCheckpointManager
from experiments.phase4_4_2.validation_suite import FixedValidationSuite

ROOT = Path(__file__).resolve().parent.parent


def _resolve_action_dims(action_spaces: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    dims: Dict[str, Dict[str, Any]] = {}
    for ag, space in action_spaces.items():
        if hasattr(space, "n"):
            dims[ag] = {"type": "Discrete", "n": int(space.n)}
        else:
            dims[ag] = {
                "type": "Box",
                "shape": list(space.shape),
                "low": space.low.tolist(),
                "high": space.high.tolist(),
            }
    return dims


class HAPPOTrainer:
    """End-to-end matched sequential HAPPO trainer."""

    def __init__(
        self,
        config: HAPPOConfig,
        output_dir: str | Path,
        checkpoint_dir: Optional[str | Path] = None,
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else (self.output_dir / "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(int(config.seed))

        models_dir = config.forecast_models_dir or str(ROOT / "experiments" / "phase4" / "forecasting" / "models")
        reward_path = config.reward_config_path
        if reward_path and not Path(reward_path).is_absolute():
            reward_path = str(ROOT / reward_path)
        self.resolved_reward_path = reward_path

        self.env = MicrogridParallelEnv(
            scenario_id=config.scenario_id,
            max_steps=config.max_steps,
            dt_minutes=config.dt_minutes,
            enable_safety_shield=config.enable_safety_shield,
            random_seed=config.seed,
            forecast_mode=config.forecast_mode,
            forecast_models_dir=models_dir,
            reward_config_path=reward_path,
            enable_adequacy_check=config.enable_adequacy_check,
            adequacy_fallback=config.adequacy_fallback_during_train,
        )

        self.obs_dims = {ag: int(self.env.observation_space(ag).shape[0]) for ag in self.env.possible_agents}
        self.state_dim = int(self.env._global_state_space.shape[0])
        self.action_dims = _resolve_action_dims(self.env._action_spaces)
        self.econ_cfg = EconomicConfig.from_yaml(str(ROOT / "experiments" / "phase4_4_4" / "configs" / "economics_v1.yaml"))

        self.agent = HAPPOAgent(
            config=config,
            obs_dims=self.obs_dims,
            action_spaces=self.env._action_spaces,
            state_dim=self.state_dim,
        )
        self.checkpoint_manager = HAPPOCheckpointManager(self.checkpoint_dir)
        self.buffer = HAPPORolloutBuffer(
            buffer_size=config.episodes_per_update * config.max_steps,
            obs_dims=self.obs_dims,
            action_spaces=self.env._action_spaces,
            global_state_dim=self.state_dim,
            device=config.device,
        )

        self.training_history: List[Dict[str, Any]] = []
        self.validation_history: List[Dict[str, Any]] = []
        self.factor_diagnostics: List[Dict[str, Any]] = []
        self.total_env_steps = 0
        self.best_val_score = -float("inf")
        self.best_checkpoint_episode = -1

    def _get_update_order(self) -> List[str]:
        if self.config.randomize_update_order:
            return [str(x) for x in self.rng.permutation(self.config.update_order)]
        return list(self.config.update_order)

    def _build_metadata(
        self,
        episode: int,
        val_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "algorithm": "HAPPO",
            "train_seed": int(self.config.seed),
            "episode": int(episode),
            "env_steps": int(self.total_env_steps),
            "agents": list(self.env.possible_agents),
            "update_order": list(self.config.update_order),
            "randomize_update_order": bool(self.config.randomize_update_order),
            "obs_dims": self.obs_dims,
            "state_dim": int(self.state_dim),
            "action_dims": self.action_dims,
            "critic_scope": "centralized_global_state",
            "architecture": {
                "actor_hidden_dims": list(self.config.actor_hidden_dims),
                "critic_hidden_dims": list(self.config.critic_hidden_dims),
                "activation": self.config.activation,
                "init_gen_bias": float(self.config.init_gen_bias),
            },
            "reward_config": str(self.config.reward_config_path),
            "env_config": {
                "scenario_id": self.config.scenario_id,
                "max_steps": int(self.config.max_steps),
                "dt_minutes": float(self.config.dt_minutes),
                "enable_safety_shield": bool(self.config.enable_safety_shield),
                "enable_adequacy_check": bool(self.config.enable_adequacy_check),
                "adequacy_fallback_during_train": bool(self.config.adequacy_fallback_during_train),
            },
            "forecast_mode": self.config.forecast_mode,
            "domain_randomization": {
                "domain_randomization": bool(self.config.domain_randomization),
                "randomize_scenarios": bool(self.config.randomize_scenarios),
                "use_curriculum": bool(self.config.use_curriculum),
            },
            "validation_metrics": dict(val_metrics),
            "best_checkpoint_episode": int(self.best_checkpoint_episode),
        }

    def train(
        self,
        total_episodes: Optional[int] = None,
        val_interval_episodes: Optional[int] = None,
        save_interval_episodes: Optional[int] = None,
    ) -> Dict[str, Any]:
        total_eps = int(total_episodes or self.config.total_episodes)
        val_interval = int(val_interval_episodes or self.config.val_interval_episodes)
        save_interval = int(save_interval_episodes or self.config.save_interval)
        episodes_per_update = int(self.config.episodes_per_update)
        t_start = time.time()

        # Initial Episode-0 fixed validation baseline (identical prior check across algorithms)
        init_val = FixedValidationSuite.evaluate_agent(
            self.agent,
            forecast_mode=self.config.forecast_mode or "realistic",
            reward_config_path=self.resolved_reward_path,
            econ_config=self.econ_cfg,
            enable_adequacy_check=self.config.enable_adequacy_check,
            adequacy_fallback=False,
            max_steps=self.config.max_steps,
        )
        init_val["episode"] = 0
        init_val["selected_best"] = True
        init_val["algorithm"] = "HAPPO"
        init_val["seed"] = int(self.config.seed)
        init_val["update_order"] = "->".join(self.config.update_order)
        self.best_val_score = float(init_val["val_score"])
        self.best_checkpoint_episode = 0
        self.validation_history.append(init_val)
        self.checkpoint_manager.save_checkpoint(
            self.agent,
            episode=0,
            metrics=init_val,
            metadata=self._build_metadata(0, init_val),
            is_best=True,
        )

        ep_idx = 0
        update_idx = 0
        while ep_idx < total_eps:
            self.buffer.reset()
            self.agent.train(True)
            batch_ep_count = min(episodes_per_update, total_eps - ep_idx)

            for _ in range(batch_ep_count):
                ep_idx += 1
                ep_start_t = time.time()
                diff = (
                    1 if ep_idx <= max(8, total_eps // 5)
                    else (2 if ep_idx <= max(20, total_eps // 2) else 3)
                ) if self.config.use_curriculum else 1
                reset_options = {
                    "difficulty": diff,
                    "randomize": bool(self.config.randomize_scenarios),
                    "domain_randomization": bool(self.config.domain_randomization),
                    "scenario_id": self.config.scenario_id,
                }
                obs, infos = self.env.reset(seed=int(self.config.seed) * 1000 + ep_idx, options=reset_options)
                global_state = infos["global_state"]

                ep_reward = 0.0
                feasible_actions = 0
                total_actions = 0
                done = False
                terms: Dict[str, bool] = {}

                while self.env.agents and not done:
                    proposed_actions, log_probs = self.agent.select_actions(obs, deterministic=False)
                    val = float(self.agent.get_value(global_state))

                    next_obs, rewards, terms, truncs, next_infos = self.env.step(proposed_actions)
                    r_step = float(rewards.get("battery", 0.0))
                    ep_reward += r_step
                    self.total_env_steps += 1

                    diag = next_infos.get("battery", {}).get("shield_diagnostics", {})
                    for _, cls_label in diag.get("action_classification", {}).items():
                        total_actions += 1
                        if cls_label == "FEASIBLE":
                            feasible_actions += 1

                    term_step = any(terms.values())
                    trunc_step = any(truncs.values())
                    done = term_step or trunc_step

                    self.buffer.add(
                        obs_dict=obs,
                        global_state=global_state,
                        proposed_actions_dict=proposed_actions,
                        log_probs_dict=log_probs,
                        reward=r_step,
                        value=val,
                        done=done,
                        truncated=trunc_step,
                    )
                    obs = next_obs
                    global_state = next_infos.get("battery", {}).get("global_state", global_state)

                next_val = 0.0 if any(terms.values()) else float(self.agent.get_value(global_state))
                finalize_episode_happo_gae(
                    buffer=self.buffer,
                    next_value=next_val,
                    gamma=self.config.gamma,
                    gae_lambda=self.config.gae_lambda,
                )

                econ = compute_episode_economics(
                    self.env.episode_history,
                    dt_hours=self.env.dt_hours,
                    config=self.econ_cfg,
                )
                self.training_history.append({
                    "episode": ep_idx,
                    "algorithm": "HAPPO",
                    "seed": int(self.config.seed),
                    "env_steps": int(self.total_env_steps),
                    "difficulty": diff,
                    "reward": round(ep_reward, 4),
                    "operating_cost_usd": econ["operating_cost_usd"],
                    "true_social_cost_usd": econ["true_social_cost_usd"],
                    "critical_ens_mwh": econ["critical_ens_mwh"],
                    "critical_reliability_pct": econ["critical_reliability_pct"],
                    "unshielded_feasible_pct": round((feasible_actions / total_actions * 100.0) if total_actions else 100.0, 2),
                    "duration_s": round(time.time() - ep_start_t, 2),
                })

            normalize_happo_advantages(self.buffer)
            update_idx += 1
            step_order = self._get_update_order()
            upd_metrics = update_happo(
                agent=self.agent,
                buffer=self.buffer,
                update_order=step_order,
                ppo_epochs=self.config.ppo_epochs,
                minibatch_size=self.config.minibatch_size,
                clip_eps=self.config.ppo_clip_eps,
                entropy_coef=self.config.entropy_coef,
                value_loss_coef=self.config.value_loss_coef,
                max_grad_norm=self.config.max_grad_norm,
                rng=self.rng,
            )
            order_str = "->".join(upd_metrics["update_order"])
            for ag_step, f_st in upd_metrics.get("factor_stats_per_agent", {}).items():
                self.factor_diagnostics.append({
                    "update_idx": update_idx,
                    "episode": ep_idx,
                    "seed": int(self.config.seed),
                    "agent": ag_step,
                    **f_st,
                })
            for row in self.training_history[-batch_ep_count:]:
                row["update_idx"] = update_idx
                row["update_order"] = order_str
                row["critic_loss"] = round(upd_metrics.get("critic_loss", 0.0), 6)
                row["actor_loss_mean"] = round(upd_metrics.get("actor_loss_mean", 0.0), 6)
                row["entropy_mean"] = round(upd_metrics.get("entropy_mean", 0.0), 6)
                row["grad_norm_mean"] = round(upd_metrics.get("grad_norm_mean", 0.0), 6)
                row["final_factor_mean"] = round(upd_metrics.get("final_factor_mean", 1.0), 6)
                row["final_factor_std"] = round(upd_metrics.get("final_factor_std", 0.0), 6)
                row["final_factor_p95"] = round(upd_metrics.get("final_factor_p95", 1.0), 6)
                row["final_factor_p99"] = round(upd_metrics.get("final_factor_p99", 1.0), 6)
                row["final_factor_max"] = round(upd_metrics.get("final_factor_max", 1.0), 6)
                row["final_factor_min"] = round(upd_metrics.get("final_factor_min", 1.0), 6)
                row["final_nonfinite_count"] = int(upd_metrics.get("final_nonfinite_count", 0))

            should_val = (ep_idx % val_interval == 0) or (ep_idx == total_eps)
            should_save = should_val or (ep_idx % save_interval == 0)
            if should_val:
                val_res = FixedValidationSuite.evaluate_agent(
                    self.agent,
                    forecast_mode=self.config.forecast_mode or "realistic",
                    reward_config_path=self.resolved_reward_path,
                    econ_config=self.econ_cfg,
                    enable_adequacy_check=self.config.enable_adequacy_check,
                    adequacy_fallback=False,
                    max_steps=self.config.max_steps,
                )
                is_best = float(val_res["val_score"]) > self.best_val_score
                if is_best:
                    self.best_val_score = float(val_res["val_score"])
                    self.best_checkpoint_episode = ep_idx
                val_res["episode"] = ep_idx
                val_res["selected_best"] = is_best
                val_res["algorithm"] = "HAPPO"
                val_res["seed"] = int(self.config.seed)
                val_res["update_order"] = order_str
                val_res["training_duration_s"] = round(time.time() - t_start, 2)
                self.validation_history.append(val_res)

                self.checkpoint_manager.save_checkpoint(
                    self.agent,
                    episode=ep_idx,
                    metrics=val_res,
                    metadata=self._build_metadata(ep_idx, val_res),
                    is_best=is_best,
                )
            elif should_save:
                latest_val = self.validation_history[-1] if self.validation_history else {}
                self.checkpoint_manager.save_checkpoint(
                    self.agent,
                    episode=ep_idx,
                    metrics=latest_val,
                    metadata=self._build_metadata(ep_idx, latest_val),
                    is_best=False,
                )

        latest_val = self.validation_history[-1] if self.validation_history else {}
        final_ckpt_path = self.checkpoint_manager.save_checkpoint(
            self.agent,
            episode=ep_idx,
            metrics=latest_val,
            metadata=self._build_metadata(ep_idx, latest_val),
            is_best=False,
            tag="final",
        )

        pd.DataFrame(self.training_history).to_csv(self.output_dir / "training_history.csv", index=False)
        pd.DataFrame(self.validation_history).to_csv(self.output_dir / "validation_history.csv", index=False)
        pd.DataFrame(self.factor_diagnostics).to_csv(self.output_dir / "happo_factor_diagnostics.csv", index=False)

        summary = {
            "algorithm": "HAPPO",
            "seed": int(self.config.seed),
            "episodes": total_eps,
            "env_steps": int(self.total_env_steps),
            "update_order": list(self.config.update_order),
            "best_val_score": float(self.best_val_score),
            "best_checkpoint_episode": int(self.best_checkpoint_episode),
            "best_checkpoint_path": str(self.checkpoint_dir / "best_model.pt"),
            "final_checkpoint_episode": int(ep_idx),
            "final_checkpoint_path": str(final_ckpt_path),
            "duration_s": round(time.time() - t_start, 2),
        }
        with open(self.output_dir / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary
