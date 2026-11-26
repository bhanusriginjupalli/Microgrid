"""
mappo/trainer.py
============================================================
MAPPO Training Orchestrator:
Executes rollout collection, GAE computation, PPO updates,
curriculum difficulty scaling, metric logging, and checkpointing.
============================================================
"""

import time
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from rl_environment.microgrid_parallel_env import MicrogridParallelEnv
from mappo.config import MAPPOConfig
from mappo.mappo_agent import MAPPOAgent
from mappo.rollout_buffer import MultiAgentRolloutBuffer
from mappo.gae import compute_gae
from mappo.ppo_update import update_mappo
from mappo.checkpoint_manager import CheckpointManager
from mappo.evaluator import MAPPOEvaluator


class MAPPOTrainer:
    """Manages the end-to-end training lifecycle of MAPPO agents."""

    def __init__(self, config: MAPPOConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Output paths
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)

        # Initialize Environment
        self.env = MicrogridParallelEnv(
            scenario_id=config.scenario_id,
            max_steps=config.max_steps,
            dt_minutes=config.dt_minutes,
            enable_safety_shield=config.enable_safety_shield,
            random_seed=config.seed,
            difficulty_level=config.difficulty_level,
            randomize_scenarios=config.randomize_scenarios,
            reward_config_path=config.reward_config_path,
            forecast_mode=config.forecast_mode,
            forecast_models_dir=config.forecast_models_dir,
        )

        obs_dims = {ag: self.env.observation_space(ag).shape[0] for ag in self.env.possible_agents}
        action_shapes = {
            "renewable": (2,),
            "battery": (1,),
            "generator": (1,),
            "load_management": (),
            "grid_resilience": (),
        }

        # Initialize MAPPO Agent Coordinator
        self.agent = MAPPOAgent(
            config=config,
            obs_dims=obs_dims,
            action_spaces=self.env._action_spaces,
            state_dim=self.env._global_state_space.shape[0],
        )

        # Initialize Rollout Buffer
        self.buffer = MultiAgentRolloutBuffer(
            agents=self.env.possible_agents,
            buffer_size=config.max_steps,
            obs_shapes={ag: (obs_dims[ag],) for ag in self.env.possible_agents},
            action_shapes=action_shapes,
            global_state_dim=self.env._global_state_space.shape[0],
            device=config.device,
        )

        self.evaluator = MAPPOEvaluator(self.agent)
        self.training_history: List[Dict[str, Any]] = []

    def _get_current_curriculum_stage(self, episode: int) -> int:
        """Determine curriculum difficulty level based on training progress."""
        if not self.config.use_curriculum:
            return self.config.difficulty_level

        for stage in self.config.curriculum_stages:
            if episode <= stage["max_episode"]:
                return stage["difficulty"]
        return 5

    def train(self, episodes: Optional[int] = None, start_episode: int = 1) -> List[Dict[str, Any]]:
        """
        Main MAPPO training loop.
        """
        total_eps = episodes or self.config.total_episodes
        print(f"[*] Commencing MAPPO Training | Range: Ep {start_episode} to {total_eps} | Device: {self.config.device}")

        best_eval_reward = -float("inf")
        start_time = time.time()

        for ep in range(start_episode, total_eps + 1):
            ep_start = time.time()

            # 1. Update Curriculum Difficulty
            current_diff = self._get_current_curriculum_stage(ep)
            should_randomize = self.config.randomize_scenarios or (current_diff >= 3 and ep % 2 == 0)

            # 2. Reset Environment with curriculum options
            reset_options = {
                "difficulty": current_diff,
                "randomize": should_randomize or getattr(self.config, "domain_randomization", False),
                "domain_randomization": getattr(self.config, "domain_randomization", False),
                "scenario_id": self.config.scenario_id,
            }
            obs, infos = self.env.reset(seed=self.config.seed + ep, options=reset_options)
            self.buffer.reset()

            ep_reward = 0.0
            global_state = infos["global_state"]

            # 3. Rollout Collection
            while self.env.agents:
                # Decentralized action proposal
                proposed_actions, log_probs = self.agent.select_actions(obs, deterministic=False)

                # Centralized state value prediction
                value = self.agent.get_value(global_state)

                # Step environment (SafetyShield runs inside step)
                next_obs, rewards, terminations, truncations, next_infos = self.env.step(proposed_actions)

                # Telemetry from history
                last_step = self.env.episode_history[-1]
                reward_step = rewards.get("battery", 0.0)
                ep_reward += reward_step

                done = any(terminations.values())
                truncated = any(truncations.values())
                shield_info = last_step.get("any_intervention", False)

                # Insert into rollout buffer
                self.buffer.insert(
                    obs=obs,
                    global_state=global_state,
                    proposed_actions=proposed_actions,
                    executed_actions={
                        "renewable": np.array([1.0 - last_step["solar_mw"] / max(1e-4, last_step["solar_available_mw"]),
                                               1.0 - last_step["wind_mw"] / max(1e-4, last_step["wind_available_mw"])]),
                        "battery": np.array([last_step["battery_mw"] / 0.25]),
                        "generator": np.array([last_step["generator_mw"] / 0.40]),
                        "load_management": 0,
                        "grid_resilience": 0 if last_step["grid_connected"] else 2,
                    },
                    log_probs=log_probs,
                    reward=reward_step,
                    value=value,
                    done=done,
                    truncated=truncated,
                    shield_info={"any_intervention": shield_info},
                )

                obs = next_obs
                infos = next_infos
                if "global_state" in infos:
                    global_state = infos["global_state"]

            # 4. Compute GAE
            last_global_state = self.env.obs_builder.build_global_state(self.env.episode_history[-1])
            next_value = self.agent.get_value(last_global_state)
            compute_gae(
                buffer=self.buffer,
                next_value=next_value,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                normalize_advantages=self.config.normalize_advantages,
            )

            # 5. Execute PPO Updates
            update_metrics = update_mappo(
                actors=self.agent.actors,
                critic=self.agent.critic,
                actor_optimizers=self.agent.actor_optimizers,
                critic_optimizer=self.agent.critic_optimizer,
                buffer=self.buffer,
                ppo_epochs=self.config.ppo_epochs,
                minibatch_size=self.config.minibatch_size,
                clip_eps=self.config.ppo_clip_eps,
                entropy_coef=self.config.entropy_coef,
                value_loss_coef=self.config.value_loss_coef,
                max_grad_norm=self.config.max_grad_norm,
            )

            # 6. Compile Episode Telemetry
            history_df = pd.DataFrame(self.env.episode_history)
            dt_h = self.env.dt_hours
            tot_crit_mwh = history_df["nominal_critical_mw"].sum() * dt_h
            served_crit_mwh = history_df["served_critical_mw"].sum() * dt_h
            shed_crit_mwh = history_df["shed_critical_mw"].sum() * dt_h
            shed_flex_mwh = history_df["shed_flexible_mw"].sum() * dt_h
            tot_fuel_l = history_df["step_fuel_l"].sum()
            interventions = int(history_df["any_intervention"].sum())
            curt_mwh = (
                (history_df["solar_available_mw"] - history_df["solar_mw"]).clip(lower=0.0).sum()
                + (history_df["wind_available_mw"] - history_df["wind_mw"]).clip(lower=0.0).sum()
            ) * dt_h

            import_mwh = history_df["grid_import_mw"].clip(lower=0.0).sum() * dt_h
            export_mwh = (-history_df["grid_import_mw"].clip(upper=0.0)).sum() * dt_h
            operating_cost = (import_mwh * 120.0) - (export_mwh * 40.0) + (tot_fuel_l * 1.50)
            rel_pct = (served_crit_mwh / tot_crit_mwh * 100.0) if tot_crit_mwh > 0 else 100.0

            ep_duration = time.time() - ep_start
            rec = {
                "episode": ep,
                "curriculum_difficulty": current_diff,
                "episode_reward": round(ep_reward, 2),
                "critical_reliability_pct": round(rel_pct, 2),
                "critical_ens_mwh": round(shed_crit_mwh, 3),
                "flexible_ens_mwh": round(shed_flex_mwh, 3),
                "operating_cost_usd": round(operating_cost, 2),
                "fuel_consumed_l": round(tot_fuel_l, 1),
                "renewable_curtailment_mwh": round(curt_mwh, 3),
                "shield_interventions": interventions,
                "critic_loss": round(update_metrics.get("critic_loss", 0.0), 4),
                "actor_loss_battery": round(update_metrics.get("actor_loss_battery", 0.0), 4),
                "entropy_battery": round(update_metrics.get("entropy_battery", 0.0), 4),
                "duration_seconds": round(ep_duration, 2),
            }
            self.training_history.append(rec)

            # Rolling reward calculation
            recent_rewards = [r["episode_reward"] for r in self.training_history[-20:]]
            rolling_avg_reward = np.mean(recent_rewards)
            rec["rolling_reward_20"] = round(rolling_avg_reward, 2)

            if ep % 10 == 0 or ep == 1:
                print(
                    f"Ep {ep:4d}/{total_eps} [Diff {current_diff}] | "
                    f"Reward: {ep_reward:7.2f} (Roll: {rolling_avg_reward:7.2f}) | "
                    f"CritRel: {rel_pct:6.2f}% | "
                    f"Cost: ${operating_cost:7.2f} | "
                    f"Interv: {interventions:2d} | "
                    f"Loss(V): {rec['critic_loss']:6.4f} | "
                    f"Time: {ep_duration:4.2f}s"
                )

            # 7. Checkpointing
            is_best = False
            if ep >= 10 and rolling_avg_reward > best_eval_reward:
                best_eval_reward = rolling_avg_reward
                is_best = True

            if ep % self.config.checkpoint_interval_episodes == 0 or is_best:
                self.checkpoint_manager.save_checkpoint(
                    agent=self.agent,
                    episode=ep,
                    metrics=rec,
                    is_best=is_best,
                )

        # Save training logs to CSV
        csv_path = self.log_dir / "mappo_training_log.csv"
        new_df = pd.DataFrame(self.training_history)
        if start_episode > 1 and csv_path.exists():
            old_df = pd.read_csv(csv_path)
            log_df = pd.concat([old_df[old_df["episode"] < start_episode], new_df], ignore_index=True)
        else:
            log_df = new_df
        log_df.to_csv(csv_path, index=False)
        print(f"\n[+] Training complete in {time.time() - start_time:.1f}s. Log saved to: {csv_path}")

        return self.training_history
