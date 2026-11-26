"""
experiments/phase4_4_4/constrained_mappo_v3.py
==============================================
GRIDGUARD AI — Phase 4.4.4 Constrained MAPPO Trainer v3

Preserves the exact MAPPOAgent checkpoint schema while adding:
  - Per-Constraint Cost Value Heads (Constraint GAE replacing unbaselined c_ret)
    when remediation is enabled.
  - Agent-localized shaped advantages for Generator (Remediation A) and Battery
    (Remediation B), eliminating cross-agent credit-assignment dilution and
    SafetyShield reward masking.
  - Strict 2x2 ablation toggle support via ablation_*.yaml configs.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "phase1_digital_twin") not in sys.path:
    sys.path.insert(0, str(ROOT / "phase1_digital_twin"))

from rl_environment.microgrid_parallel_env import MicrogridParallelEnv
from rl_environment.economic_accounting import compute_episode_economics, EconomicConfig
from mappo.config import MAPPOConfig
from mappo.mappo_agent import MAPPOAgent
from mappo.checkpoint_manager import CheckpointManager
from experiments.phase4_4_2.validation_suite import FixedValidationSuite
from experiments.phase4_4_2.constrained_mappo_v2 import (
    _compute_episode_gae,
    SafetyConstraintEvaluatorV2,
    AdaptiveLagrangeManagerV2,
)
from experiments.phase4_4_4.remediation_v3 import (
    RemediationConfig,
    classify_generator_proposal,
    compute_battery_remediation_signals,
)


class AuxiliaryValueHead(nn.Module):
    """Trainer-only scalar/vector value baseline (not stored in MAPPOAgent checkpoints)."""

    def __init__(self, state_dim: int, out_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SafetyConstraintEvaluatorV3(SafetyConstraintEvaluatorV2):
    """Condition-aware cost evaluator extending V2 cleanly for Phase 4.4.4."""

    def __init__(self, config_path: str, rem_cfg: RemediationConfig):
        super().__init__(config_path=config_path)
        self.rem_cfg = rem_cfg

    def evaluate_step_costs_v3(
        self,
        state_dict: Dict[str, Any],
        raw_actions: Dict[str, Any],
        adequacy_diagnostics: Optional[Dict[str, Any]],
        gen_diag: Dict[str, Any],
        bat_diag: Dict[str, Any],
    ) -> Dict[str, float]:
        costs = super().evaluate_step_costs(
            state_dict=state_dict,
            raw_actions=raw_actions,
            adequacy_diagnostics=adequacy_diagnostics,
        )

        # Override generator cost when Remediation A active: zero when necessary or min-up hold
        if self.rem_cfg.remediation_gen:
            raw_gen = float(np.asarray(raw_actions.get("generator", 0.0)).flatten()[0])
            c_gen = 0.0
            if gen_diag["unnecessary_dispatch"]:
                c_gen += 1.0 + 4.0 * max(0.0, raw_gen - 0.125)
            gen_mw = float(state_dict.get("generator_mw", 0.0))
            if gen_mw > 0.40:
                c_gen += (gen_mw - 0.40) * 10.0
            costs["gen"] = float(np.clip(c_gen, 0.0, 10.0))

        # Augment battery reserve/deficit cost when Remediation B active
        if self.rem_cfg.remediation_bat:
            c_bat = float(costs["bat"])
            if bat_diag["deficit_penalty"] > 0.0:
                c_bat += min(3.0, bat_diag["deficit_penalty"])
            if bat_diag["reserve_breach_attempt"]:
                c_bat += 1.5
            costs["bat"] = float(np.clip(c_bat, 0.0, 10.0))

        return costs


class ConstrainedMAPPOTrainerV3:
    """
    Phase 4.4.4 Trainer supporting exact 2x2 factorial ablation:
      Baseline (False, False), Gen-only (True, False),
      Bat-only (False, True), Combined (True, True).
    """

    def __init__(
        self,
        config: MAPPOConfig,
        ablation_cfg_path: str,
        econ_cfg_path: str,
        adaptive_multipliers: bool = True,
        use_tail_risk: bool = False,
        enable_adequacy_fallback_during_train: bool = False,
    ):
        self.config = config
        self.cfg_path = str(ablation_cfg_path)
        with open(self.cfg_path, "r", encoding="utf-8") as f:
            self.v3_cfg = yaml.safe_load(f)

        self.rem_cfg = RemediationConfig.from_yaml(self.cfg_path)
        self.ppo_cfg = self.v3_cfg["ppo_update"]
        self.tail_cfg = self.v3_cfg["tail_risk_cvar"]
        self.adaptive_multipliers = adaptive_multipliers
        self.use_tail_risk = use_tail_risk
        self.enable_adequacy_fallback_during_train = enable_adequacy_fallback_during_train

        self.econ_cfg = EconomicConfig.from_yaml(econ_cfg_path)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.log_dir = Path(config.log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.env = MicrogridParallelEnv(
            scenario_id=config.scenario_id,
            forecast_mode=config.forecast_mode,
            forecast_models_dir=config.forecast_models_dir,
            max_steps=config.max_steps,
            dt_minutes=config.dt_minutes,
            enable_safety_shield=config.enable_safety_shield,
            reward_config_path=config.reward_config_path,
            enable_adequacy_check=True,
            adequacy_fallback=enable_adequacy_fallback_during_train,
        )

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        obs_dims = {ag: self.env.observation_space(ag).shape[0] for ag in self.env.possible_agents}
        state_dim = self.env._global_state_space.shape[0]
        self.agent = MAPPOAgent(
            config=config,
            obs_dims=obs_dims,
            action_spaces=self.env._action_spaces,
            state_dim=state_dim,
        )

        if "generator" in self.agent.actors and hasattr(self.agent.actors["generator"], "mean_head"):
            nn.init.constant_(self.agent.actors["generator"].mean_head.bias, float(self.rem_cfg.init_gen_bias))

        for ag, opt in self.agent.actor_optimizers.items():
            for pg in opt.param_groups:
                pg["lr"] = float(self.ppo_cfg["actor_lr"])
        for pg in self.agent.critic_optimizer.param_groups:
            pg["lr"] = float(self.ppo_cfg["critic_lr"])

        self.checkpoint_manager = CheckpointManager(checkpoint_dir=str(self.checkpoint_dir))
        self.cost_evaluator = SafetyConstraintEvaluatorV3(self.cfg_path, self.rem_cfg)
        self.lagrange_mgr = AdaptiveLagrangeManagerV2(self.cfg_path, adaptive=adaptive_multipliers)

        # Trainer-only auxiliary critics (keep MAPPOAgent checkpoints backward-compatible)
        critic_lr = float(self.ppo_cfg["critic_lr"])
        self.cost_critic: Optional[AuxiliaryValueHead] = None
        self.cost_critic_opt: Optional[torch.optim.Adam] = None
        if self.rem_cfg.use_cost_critics:
            self.cost_critic = AuxiliaryValueHead(state_dim, out_dim=len(self.lagrange_mgr.cost_keys))
            self.cost_critic_opt = torch.optim.Adam(self.cost_critic.parameters(), lr=critic_lr)

        self.agent_critics: Dict[str, AuxiliaryValueHead] = {}
        self.agent_critic_opts: Dict[str, torch.optim.Adam] = {}
        if self.rem_cfg.remediation_gen:
            self.agent_critics["generator"] = AuxiliaryValueHead(state_dim, out_dim=1)
            self.agent_critic_opts["generator"] = torch.optim.Adam(self.agent_critics["generator"].parameters(), lr=critic_lr)
        if self.rem_cfg.remediation_bat:
            self.agent_critics["battery"] = AuxiliaryValueHead(state_dim, out_dim=1)
            self.agent_critic_opts["battery"] = torch.optim.Adam(self.agent_critics["battery"].parameters(), lr=critic_lr)

        self.training_history: List[Dict[str, Any]] = []
        self.multiplier_history: List[Dict[str, Any]] = []
        self.validation_history: List[Dict[str, Any]] = []
        self.best_val_score = -float("inf")
        self.best_checkpoint_episode = -1

    def train(self, total_episodes: int = 100, val_interval_episodes: int = 20) -> Dict[str, Any]:
        episodes_per_update = int(self.ppo_cfg["episodes_per_update"])
        ppo_epochs = int(self.ppo_cfg["ppo_epochs"])
        minibatch_size = int(self.ppo_cfg["minibatch_size"])
        gamma = float(self.ppo_cfg["gamma"])
        gae_lambda = float(self.ppo_cfg["gae_lambda"])
        clip_eps = float(self.ppo_cfg["clip_eps"])
        entropy_coef = float(self.ppo_cfg["entropy_coef"])
        max_grad_norm = float(self.ppo_cfg["max_grad_norm"])

        print(
            f"[*] Phase 4.4.4 {self.rem_cfg.experiment_name} | Seed={self.config.seed} "
            f"| GenRem={self.rem_cfg.remediation_gen} | BatRem={self.rem_cfg.remediation_bat} "
            f"| Episodes={total_episodes}",
            flush=True,
        )

        init_val = FixedValidationSuite.evaluate_agent(
            self.agent,
            forecast_mode=self.config.forecast_mode or "realistic",
            reward_config_path=self.config.reward_config_path,
            econ_config=self.econ_cfg,
            enable_adequacy_check=True,
            adequacy_fallback=False,
            max_steps=self.config.max_steps,
        )
        init_val["episode"] = 0
        init_val["selected_best"] = True
        self.best_val_score = init_val["val_score"]
        self.best_checkpoint_episode = 0
        self.validation_history.append(init_val)
        self.checkpoint_manager.save_checkpoint(self.agent, episode=0, metrics=init_val, is_best=True)

        ep_idx = 0
        update_idx = 0

        while ep_idx < total_episodes:
            batch_obs: Dict[str, List[np.ndarray]] = {ag: [] for ag in self.env.possible_agents}
            batch_acts: Dict[str, List[Any]] = {ag: [] for ag in self.env.possible_agents}
            batch_log_probs: Dict[str, List[float]] = {ag: [] for ag in self.env.possible_agents}
            batch_states: List[np.ndarray] = []
            batch_returns: List[np.ndarray] = []
            batch_raw_rew_adv: List[np.ndarray] = []
            batch_raw_cost_advs: Dict[str, List[np.ndarray]] = {k: [] for k in self.lagrange_mgr.cost_keys}
            batch_cost_returns: Dict[str, List[np.ndarray]] = {k: [] for k in self.lagrange_mgr.cost_keys}
            batch_agent_returns: Dict[str, List[np.ndarray]] = {ag: [] for ag in self.agent_critics}
            batch_agent_rew_adv: Dict[str, List[np.ndarray]] = {ag: [] for ag in self.agent_critics}
            batch_ep_mean_costs: Dict[str, List[float]] = {k: [] for k in self.lagrange_mgr.cost_keys}

            batch_ep_count = min(episodes_per_update, total_episodes - ep_idx)

            for _ in range(batch_ep_count):
                ep_idx += 1
                ep_start = time.time()
                diff = 1 if ep_idx <= max(8, total_episodes // 5) else (2 if ep_idx <= max(20, total_episodes // 2) else 3)
                reset_options = {
                    "difficulty": diff,
                    "randomize": True,
                    "domain_randomization": True,
                    "scenario_id": self.config.scenario_id,
                }
                obs, infos = self.env.reset(seed=self.config.seed * 1000 + ep_idx, options=reset_options)
                global_state = infos["global_state"]

                ep_rewards: List[float] = []
                ep_values: List[float] = []
                ep_terms: List[bool] = []
                ep_costs: Dict[str, List[float]] = {k: [] for k in self.lagrange_mgr.cost_keys}
                ep_cost_vals: Dict[str, List[float]] = {k: [] for k in self.lagrange_mgr.cost_keys}
                ep_agent_rewards: Dict[str, List[float]] = {ag: [] for ag in self.agent_critics}
                ep_agent_vals: Dict[str, List[float]] = {ag: [] for ag in self.agent_critics}

                unneeded_gen_steps = 0
                necessary_gen_steps = 0
                deficit_bat_steps = 0
                genuine_deficit_steps = 0
                inaction_steps = 0
                feasible_actions = 0
                total_actions = 0
                ep_reward = 0.0
                done = False

                while self.env.agents and not done:
                    pre_gen_state = str(self.env.generator.state.value)
                    pre_gen_steps = int(self.env.generator.steps_in_state)
                    prev_bat_energy = float(self.env.battery.energy_mwh)

                    proposed_actions, log_probs = self.agent.select_actions(obs, deterministic=False)
                    val = float(self.agent.get_value(global_state))

                    with torch.no_grad():
                        gs_t = torch.as_tensor(global_state, dtype=torch.float32).unsqueeze(0)
                        if self.cost_critic is not None:
                            c_vals_step = self.cost_critic(gs_t).squeeze(0).numpy()
                        else:
                            c_vals_step = None
                        ag_vals_step = {
                            ag: float(cr(gs_t).item()) for ag, cr in self.agent_critics.items()
                        }

                    next_obs, rewards, terms, truncs, next_infos = self.env.step(proposed_actions)
                    last_st = self.env.episode_history[-1]
                    diag = next_infos.get("battery", {}).get("shield_diagnostics", {})
                    adeq = next_infos.get("battery", {}).get("adequacy_diagnostics", {})

                    for ag, st_cls in diag.get("action_classification", {}).items():
                        total_actions += 1
                        if st_cls == "FEASIBLE":
                            feasible_actions += 1
                    if adeq.get("inaction_detected", False):
                        inaction_steps += 1

                    raw_gen = float(np.asarray(proposed_actions["generator"]).flatten()[0])
                    raw_bat = float(np.asarray(proposed_actions["battery"]).flatten()[0])

                    gen_diag = classify_generator_proposal(
                        state_dict=last_st,
                        raw_gen_frac=raw_gen,
                        shield_diag=diag,
                        adeq_diag=adeq,
                        pre_gen_state=pre_gen_state,
                        pre_gen_steps_in_state=pre_gen_steps,
                        rem_cfg=self.rem_cfg,
                    )
                    bat_diag = compute_battery_remediation_signals(
                        state_dict=last_st,
                        prev_bat_energy_mwh=prev_bat_energy,
                        raw_bat_cmd=raw_bat,
                        shield_diag=diag,
                        adeq_diag=adeq,
                        dt_hours=self.env.dt_hours,
                        rem_cfg=self.rem_cfg,
                    )

                    if gen_diag["unnecessary_dispatch"]:
                        unneeded_gen_steps += 1
                    if gen_diag["necessary_dispatch"]:
                        necessary_gen_steps += 1
                    if bat_diag["genuine_deficit"]:
                        genuine_deficit_steps += 1
                        if bat_diag["battery_discharged_deficit"]:
                            deficit_bat_steps += 1

                    step_costs = self.cost_evaluator.evaluate_step_costs_v3(
                        state_dict=last_st,
                        raw_actions=proposed_actions,
                        adequacy_diagnostics=adeq,
                        gen_diag=gen_diag,
                        bat_diag=bat_diag,
                    )

                    r_step = float(rewards.get("battery", 0.0))
                    ep_reward += r_step

                    for ag in self.env.possible_agents:
                        batch_obs[ag].append(obs[ag])
                        batch_acts[ag].append(proposed_actions[ag])
                        batch_log_probs[ag].append(float(log_probs[ag]))
                    batch_states.append(global_state)

                    ep_rewards.append(r_step)
                    ep_values.append(val)
                    term_step = any(terms.values())
                    done_step = term_step or any(truncs.values())
                    ep_terms.append(term_step)

                    for k_i, k in enumerate(self.lagrange_mgr.cost_keys):
                        ep_costs[k].append(step_costs[k])
                        if c_vals_step is not None:
                            ep_cost_vals[k].append(float(c_vals_step[k_i]))

                    if "generator" in self.agent_critics:
                        ep_agent_rewards["generator"].append(r_step + gen_diag["gen_reward_delta"])
                        ep_agent_vals["generator"].append(ag_vals_step["generator"])
                    if "battery" in self.agent_critics:
                        ep_agent_rewards["battery"].append(r_step + bat_diag["bat_reward_delta"])
                        ep_agent_vals["battery"].append(ag_vals_step["battery"])

                    obs = next_obs
                    global_state = next_infos.get("battery", {}).get("global_state", global_state)
                    done = done_step

                # GAE calculations
                T_ep = len(ep_rewards)
                with torch.no_grad():
                    gs_end_t = torch.as_tensor(global_state, dtype=torch.float32).unsqueeze(0)
                    last_val = 0.0 if any(terms.values()) else float(self.agent.get_value(global_state))
                    last_c_vals = (
                        np.zeros(len(self.lagrange_mgr.cost_keys), dtype=np.float32)
                        if any(terms.values()) or self.cost_critic is None
                        else self.cost_critic(gs_end_t).squeeze(0).numpy()
                    )
                    last_ag_vals = {
                        ag: 0.0 if any(terms.values()) else float(cr(gs_end_t).item())
                        for ag, cr in self.agent_critics.items()
                    }

                rew_adv_ep, ret_ep = _compute_episode_gae(
                    rewards=np.asarray(ep_rewards, dtype=np.float32),
                    values=np.asarray(ep_values, dtype=np.float32),
                    terminated_flags=np.asarray(ep_terms, dtype=np.float32),
                    last_value=last_val,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                )
                batch_returns.append(ret_ep)
                batch_raw_rew_adv.append(rew_adv_ep)

                for ag in self.agent_critics:
                    ag_adv, ag_ret = _compute_episode_gae(
                        rewards=np.asarray(ep_agent_rewards[ag], dtype=np.float32),
                        values=np.asarray(ep_agent_vals[ag], dtype=np.float32),
                        terminated_flags=np.asarray(ep_terms, dtype=np.float32),
                        last_value=last_ag_vals[ag],
                        gamma=gamma,
                        gae_lambda=gae_lambda,
                    )
                    batch_agent_rew_adv[ag].append(ag_adv)
                    batch_agent_returns[ag].append(ag_ret)

                for k_i, k in enumerate(self.lagrange_mgr.cost_keys):
                    c_seq = np.asarray(ep_costs[k], dtype=np.float32)
                    if self.cost_critic is not None:
                        c_adv, c_ret = _compute_episode_gae(
                            rewards=c_seq,
                            values=np.asarray(ep_cost_vals[k], dtype=np.float32),
                            terminated_flags=np.asarray(ep_terms, dtype=np.float32),
                            last_value=float(last_c_vals[k_i]),
                            gamma=gamma,
                            gae_lambda=gae_lambda,
                        )
                    else:
                        c_ret = np.zeros(T_ep, dtype=np.float32)
                        run_c = 0.0
                        for t_i in reversed(range(T_ep)):
                            run_c = float(c_seq[t_i]) + gamma * run_c
                            c_ret[t_i] = run_c
                        c_adv = c_ret
                    batch_raw_cost_advs[k].append(c_adv)
                    batch_cost_returns[k].append(c_ret)
                    batch_ep_mean_costs[k].append(float(np.mean(c_seq)))

                econ = compute_episode_economics(self.env.episode_history, dt_hours=self.env.dt_hours, config=self.econ_cfg)
                self.training_history.append({
                    "episode": ep_idx,
                    "experiment": self.rem_cfg.experiment_name,
                    "seed": self.config.seed,
                    "reward": round(ep_reward, 2),
                    "operating_cost_usd": econ["operating_cost_usd"],
                    "true_social_cost_usd": econ["true_social_cost_usd"],
                    "critical_ens_mwh": econ["critical_ens_mwh"],
                    "critical_reliability_pct": econ["critical_reliability_pct"],
                    "feasible_pct": round((feasible_actions / total_actions * 100.0) if total_actions else 100.0, 2),
                    "unneeded_gen_steps": unneeded_gen_steps,
                    "necessary_gen_steps": necessary_gen_steps,
                    "deficit_bat_dis_pct": round((deficit_bat_steps / genuine_deficit_steps * 100.0) if genuine_deficit_steps > 0 else 100.0, 2),
                    "inaction_steps": inaction_steps,
                    "duration_s": round(time.time() - ep_start, 2),
                })

            # Batch PPO update
            update_idx += 1
            flat_returns = np.concatenate(batch_returns, axis=0)
            flat_raw_rew_adv = np.concatenate(batch_raw_rew_adv, axis=0)
            flat_raw_cost_advs = {k: np.concatenate(batch_raw_cost_advs[k], axis=0) for k in self.lagrange_mgr.cost_keys}
            flat_cost_returns = np.stack(
                [np.concatenate(batch_cost_returns[k], axis=0) for k in self.lagrange_mgr.cost_keys],
                axis=-1,
            )

            # Global Lagrangian advantages (for un-remediated agents)
            lag_adv_global, adv_diag = self.lagrange_mgr.compute_lagrangian_advantages(
                raw_reward_adv=flat_raw_rew_adv,
                raw_cost_advs=flat_raw_cost_advs,
                episode_weights=None,
            )

            # Per-agent advantages: remediated agents use their localized shaped advantage + normalized relevant constraints
            agent_adv_tensors: Dict[str, torch.Tensor] = {}
            for ag in self.env.possible_agents:
                if ag in self.agent_critics:
                    raw_ag_adv = np.concatenate(batch_agent_rew_adv[ag], axis=0)
                    # Focus cost constraints cleanly: generator uses {gen, inaction, crit}; battery uses {bat, inaction, crit}
                    masked_costs = dict(flat_raw_cost_advs)
                    if ag == "generator" and self.rem_cfg.remediation_gen:
                        for ck in ("bat", "volt", "switch"):
                            masked_costs[ck] = np.zeros_like(masked_costs[ck])
                    elif ag == "battery" and self.rem_cfg.remediation_bat:
                        for ck in ("gen", "volt", "switch"):
                            masked_costs[ck] = np.zeros_like(masked_costs[ck])
                    ag_lag, _ = self.lagrange_mgr.compute_lagrangian_advantages(
                        raw_reward_adv=raw_ag_adv,
                        raw_cost_advs=masked_costs,
                        episode_weights=None,
                    )
                    agent_adv_tensors[ag] = torch.tensor(ag_lag, dtype=torch.float32)
                else:
                    agent_adv_tensors[ag] = torch.tensor(lag_adv_global, dtype=torch.float32)

            N_batch = len(lag_adv_global)
            flat_states = np.asarray(batch_states, dtype=np.float32)
            state_tensor = torch.tensor(flat_states, dtype=torch.float32)
            ret_tensor = torch.tensor(flat_returns, dtype=torch.float32).unsqueeze(-1)
            cost_ret_tensor = torch.tensor(flat_cost_returns, dtype=torch.float32)

            agent_ret_tensors = {
                ag: torch.tensor(np.concatenate(batch_agent_returns[ag], axis=0), dtype=torch.float32).unsqueeze(-1)
                for ag in self.agent_critics
            }

            actor_tensors = {}
            for ag in self.env.possible_agents:
                act_arr = np.asarray(batch_acts[ag])
                act_dtype = torch.long if ag in ("load_management", "grid_resilience") else torch.float32
                actor_tensors[ag] = {
                    "obs": torch.tensor(np.asarray(batch_obs[ag], dtype=np.float32), dtype=torch.float32),
                    "acts": torch.tensor(act_arr, dtype=act_dtype),
                    "old_lp": torch.tensor(np.asarray(batch_log_probs[ag], dtype=np.float32), dtype=torch.float32),
                }

            last_v_loss = 0.0
            for _ in range(ppo_epochs):
                perm = np.random.permutation(N_batch)
                for start in range(0, N_batch, minibatch_size):
                    mb = perm[start : start + minibatch_size]
                    if len(mb) < 8:
                        continue
                    mb_idx = torch.tensor(mb, dtype=torch.long)

                    for ag in self.env.possible_agents:
                        actor = self.agent.actors[ag]
                        opt = self.agent.actor_optimizers[ag]
                        new_lp, entropy = actor.evaluate_actions(
                            actor_tensors[ag]["obs"][mb_idx],
                            actor_tensors[ag]["acts"][mb_idx],
                        )
                        ratio = torch.exp(new_lp - actor_tensors[ag]["old_lp"][mb_idx])
                        mb_adv = agent_adv_tensors[ag][mb_idx]
                        surr1 = ratio * mb_adv
                        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                        loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy.mean()
                        opt.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                        opt.step()

                    pred_v = self.agent.critic(state_tensor[mb_idx])
                    v_loss = nn.functional.mse_loss(pred_v, ret_tensor[mb_idx])
                    self.agent.critic_optimizer.zero_grad()
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.critic.parameters(), max_grad_norm)
                    self.agent.critic_optimizer.step()
                    last_v_loss = float(v_loss.item())

                    if self.cost_critic is not None and self.cost_critic_opt is not None:
                        c_pred = self.cost_critic(state_tensor[mb_idx])
                        c_loss = nn.functional.mse_loss(c_pred, cost_ret_tensor[mb_idx])
                        self.cost_critic_opt.zero_grad()
                        c_loss.backward()
                        nn.utils.clip_grad_norm_(self.cost_critic.parameters(), max_grad_norm)
                        self.cost_critic_opt.step()

                    for ag, cr in self.agent_critics.items():
                        ag_pred = cr(state_tensor[mb_idx])
                        ag_loss = nn.functional.mse_loss(ag_pred, agent_ret_tensors[ag][mb_idx])
                        self.agent_critic_opts[ag].zero_grad()
                        ag_loss.backward()
                        nn.utils.clip_grad_norm_(cr.parameters(), max_grad_norm)
                        self.agent_critic_opts[ag].step()

            mean_batch_costs = {k: float(np.mean(batch_ep_mean_costs[k])) for k in self.lagrange_mgr.cost_keys}
            new_lams = self.lagrange_mgr.update_multipliers(mean_batch_costs)

            self.multiplier_history.append({
                "update": update_idx,
                "episode": ep_idx,
                "experiment": self.rem_cfg.experiment_name,
                "seed": self.config.seed,
                "critic_loss": round(last_v_loss, 4),
                **{f"mean_cost_{k}": round(mean_batch_costs[k], 4) for k in self.lagrange_mgr.cost_keys},
                **{k: round(v, 4) for k, v in new_lams.items()},
            })

            if ep_idx % val_interval_episodes == 0 or ep_idx >= total_episodes:
                val_res = FixedValidationSuite.evaluate_agent(
                    self.agent,
                    forecast_mode=self.config.forecast_mode or "realistic",
                    reward_config_path=self.config.reward_config_path,
                    econ_config=self.econ_cfg,
                    enable_adequacy_check=True,
                    adequacy_fallback=False,
                    max_steps=self.config.max_steps,
                )
                is_best = val_res["val_score"] > self.best_val_score
                if is_best:
                    self.best_val_score = val_res["val_score"]
                    self.best_checkpoint_episode = ep_idx

                val_res["episode"] = ep_idx
                val_res["selected_best"] = bool(is_best)
                val_res["experiment"] = self.rem_cfg.experiment_name
                val_res["seed"] = self.config.seed
                self.validation_history.append(val_res)

                self.checkpoint_manager.save_checkpoint(
                    agent=self.agent,
                    episode=ep_idx,
                    metrics=val_res,
                    is_best=is_best,
                )
                print(
                    f"  [{self.rem_cfg.experiment_name} s{self.config.seed} Ep {ep_idx:03d}] "
                    f"Score={val_res['val_score']:+.1f} | Social=${val_res['val_true_social_cost_usd']:.1f} | "
                    f"Rel={val_res['val_critical_reliability_pct']:.2f}% | Best={'YES' if is_best else 'no'}",
                    flush=True,
                )

        pd.DataFrame(self.training_history).to_csv(self.log_dir / "training_episodes.csv", index=False)
        pd.DataFrame(self.multiplier_history).to_csv(self.log_dir / "multiplier_advantages_history.csv", index=False)
        pd.DataFrame(self.validation_history).to_csv(self.log_dir / "validation_selection_history.csv", index=False)

        return {
            "experiment": self.rem_cfg.experiment_name,
            "seed": self.config.seed,
            "best_checkpoint_episode": self.best_checkpoint_episode,
            "best_val_score": self.best_val_score,
            "total_episodes": ep_idx,
            "final_multipliers": self.lagrange_mgr.get_multipliers(),
        }
