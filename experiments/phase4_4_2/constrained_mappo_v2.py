"""
experiments/phase4_4_2/constrained_mappo_v2.py
==============================================
GRIDGUARD AI — Phase 4.4.2: Remediated Constrained Robust Forecast-Aware MAPPO

Addresses every algorithmic and validation flaw identified in Phase 4.4.1:
  1. Corrected Constraint Advantage Normalization:
       Normalizes each raw constraint advantage A_k^C independently BEFORE
       applying Lagrange multipliers lambda_k:
         A_lagrange = A_norm^R - sum_k lambda_k * A_norm_k^C
       This ensures lambda_k = 0 completely removes constraint k, and increasing
       lambda_k strictly increases constraint k's penalty slope.
  2. Anti-Inaction Constraint Signal (`inaction`):
       Explicitly penalizes avoidable supply shortfalls when feasible battery or
       generator resources sit unused during islanded deficits.
  3. Multi-Epoch Mini-Batch PPO Optimization over Rollout Batches:
       Replaces single-step episode updates with proper GAE + clipped PPO epochs.
  4. Episodic Rockafellar-Uryasev Tail-Risk Weighting (optional):
       Weights worst-quantile episodes by true social cost excess (Z_e - VaR)+
       rather than multiplying isolated exogenous outage steps by 11x.
  5. Fixed Validation Suite Checkpoint Selection:
       Selects best checkpoints strictly on FixedValidationSuite multi-objective
       score (True Social Cost, Critical ENS, Reliability, Constraint Violations).
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


def _compute_episode_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated_flags: np.ndarray,
    last_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute unnormalized GAE advantages and returns for a single episode."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae_lam = 0.0
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else float(values[t + 1])
        non_terminal = 1.0 - float(terminated_flags[t])
        delta = float(rewards[t]) + gamma * next_v * non_terminal - float(values[t])
        last_gae_lam = delta + gamma * gae_lambda * non_terminal * last_gae_lam
        advantages[t] = last_gae_lam
    returns = advantages + values
    return advantages, returns


DEFAULT_CONSTRAINED_V2_YAML = (
    Path(__file__).resolve().parent / "configs" / "constrained_v2.yaml"
)


class SafetyConstraintEvaluatorV2:
    """
    Evaluates 6 normalized constraint cost signals c_k(t) in [0.0, 10.0]:
      - crit:     Critical ENS (MW)
      - inaction: Avoidable supply-demand shortfall while feasible assets unused
      - bat:      Battery hard SOC limits & grid-connected reserve breaches
      - gen:      Unnecessary grid-connected generator run / over-rating
      - volt:     Bus voltage excursions outside [0.95, 1.05] pu or divergence
      - switch:   Invalid PCC reconnection on dead line or without phase sync
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = str(config_path or DEFAULT_CONSTRAINED_V2_YAML)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    def evaluate_step_costs(
        self,
        state_dict: Dict[str, Any],
        raw_actions: Dict[str, Any],
        adequacy_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        adeq = adequacy_diagnostics or {}
        costs: Dict[str, float] = {}

        # 1. Critical ENS Cost
        shed_crit = float(state_dict.get("shed_critical_mw", 0.0))
        costs["crit"] = float(np.clip((shed_crit / 0.10) if shed_crit > 1e-4 else 0.0, 0.0, 10.0))

        # 2. Avoidable Passive Inaction Cost (Islanded / Deficit Starvation)
        avoidable_shortfall = float(adeq.get("avoidable_shortfall_mw", state_dict.get("avoidable_shortfall_mw", 0.0)))
        inaction_flag = bool(adeq.get("inaction_detected", state_dict.get("inaction_detected", False)))
        c_inaction = (avoidable_shortfall / 0.10)
        if inaction_flag:
            c_inaction += 1.0
        if bool(adeq.get("can_start_generator", False)) and not state_dict.get("grid_connected", True):
            raw_gen = float(np.asarray(raw_actions.get("generator", 0.0)).flatten()[0])
            if avoidable_shortfall > 0.02 and raw_gen <= 0.04:
                c_inaction += 1.5
        costs["inaction"] = float(np.clip(c_inaction, 0.0, 10.0))

        # 3. Battery SOC & Reserve Cost
        soc = float(state_dict.get("battery_soc", 80.0))
        c_bat = 0.0
        if soc < 20.0:
            c_bat += (20.0 - soc) * 0.5
        elif soc > 95.0:
            c_bat += (soc - 95.0) * 0.5

        grid_conn = bool(state_dict.get("grid_connected", True))
        is_storm = bool(state_dict.get("is_storm_warning", False))
        if soc < 30.0 and grid_conn and not is_storm:
            c_bat += (30.0 - soc) * 0.2

        raw_bat = float(np.asarray(raw_actions.get("battery", 0.0)).flatten()[0])
        if soc <= 30.1 and raw_bat < -0.05 and grid_conn and not is_storm:
            c_bat += abs(raw_bat) * 0.5
        costs["bat"] = float(np.clip(c_bat, 0.0, 10.0))

        # 4. Generator Operating Limit Cost
        c_gen = 0.0
        raw_gen = float(np.asarray(raw_actions.get("generator", 0.0)).flatten()[0])
        if grid_conn and not is_storm and raw_gen > 0.05:
            c_gen += raw_gen * 2.0
        gen_mw = float(state_dict.get("generator_mw", 0.0))
        if gen_mw > 0.40:
            c_gen += (gen_mw - 0.40) * 10.0
        costs["gen"] = float(np.clip(c_gen, 0.0, 10.0))

        # 5. Bus Voltage & Power Flow Cost
        pcc_v = float(state_dict.get("pcc_voltage_pu", 1.0))
        c_volt = 0.0
        if pcc_v < 0.95:
            c_volt += (0.95 - pcc_v) * 50.0
        elif pcc_v > 1.05:
            c_volt += (pcc_v - 1.05) * 50.0
        if not state_dict.get("converged", True):
            c_volt += 2.0
        costs["volt"] = float(np.clip(c_volt, 0.0, 10.0))

        # 6. Invalid Switching & Sync Cost
        c_switch = 0.0
        raw_grid = int(raw_actions.get("grid_resilience", 0))
        grid_avail = bool(state_dict.get("grid_available", True))
        sync_ver = bool(state_dict.get("sync_verified", True))
        if not grid_avail and raw_grid in (0, 3):
            c_switch += 3.0
        elif grid_avail and not grid_conn and raw_grid == 3 and not sync_ver:
            c_switch += 2.0
        costs["switch"] = float(np.clip(c_switch, 0.0, 10.0))

        return costs


class AdaptiveLagrangeManagerV2:
    """
    Corrected Primal-Dual Lagrange Manager.

    Crucially, raw constraint advantages are normalized per-constraint BEFORE
    multiplying by lambda_k, guaranteeing:
      - lambda_k = 0 -> exact 0 contribution from constraint k
      - increasing lambda_k strictly scales constraint k's penalty gradient
      - non-negative bounded dual updates: lambda_k <- clip(lambda_k + alpha_k*(C_k - d_k), 0, max)
    """

    def __init__(self, config_path: Optional[str] = None, adaptive: bool = True):
        self.adaptive = adaptive
        cfg_path = str(config_path or DEFAULT_CONSTRAINED_V2_YAML)
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)["constraints"]

        self.cost_keys: List[str] = list(cfg["cost_keys"])
        self.d = {k: float(cfg["cost_limits"][k]) for k in self.cost_keys}
        self.lambdas = {k: float(cfg["multipliers_init"][k]) for k in self.cost_keys}
        self.alphas = {k: float(cfg["multipliers_lr"][k]) for k in self.cost_keys}
        self.max_lambdas = {k: float(cfg["multipliers_max"][k]) for k in self.cost_keys}
        self.history: List[Dict[str, float]] = []

    def update_multipliers(self, mean_costs: Dict[str, float]) -> Dict[str, float]:
        """Dual subgradient ascent step with non-negative bounded projection."""
        if self.adaptive:
            for k in self.cost_keys:
                c_bar = float(mean_costs.get(k, 0.0))
                diff = c_bar - self.d[k]
                new_lam = self.lambdas[k] + self.alphas[k] * diff
                if not np.isfinite(new_lam):
                    raise ValueError(f"Non-finite multiplier update for {k}: {new_lam}")
                self.lambdas[k] = float(np.clip(new_lam, 0.0, self.max_lambdas[k]))

        rec = {f"lambda_{k}": self.lambdas[k] for k in self.cost_keys}
        self.history.append(rec)
        return rec

    def get_multipliers(self) -> Dict[str, float]:
        return {f"lambda_{k}": self.lambdas[k] for k in self.cost_keys}

    def compute_lagrangian_advantages(
        self,
        raw_reward_adv: np.ndarray,
        raw_cost_advs: Dict[str, np.ndarray],
        episode_weights: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute effective Lagrangian policy advantages with pre-multiplier
        per-constraint normalization.
        """
        r_arr = np.asarray(raw_reward_adv, dtype=np.float32)
        if not np.all(np.isfinite(r_arr)):
            raise ValueError("Non-finite values in raw_reward_adv.")

        r_centered = r_arr - np.mean(r_arr)
        r_std = float(np.std(r_arr))
        if len(r_arr) > 1 and r_std > 1e-6:
            norm_rew_adv = r_centered / (r_std + 1e-8)
        else:
            norm_rew_adv = r_centered

        norm_cost_advs: Dict[str, np.ndarray] = {}
        weighted_cost_terms: Dict[str, np.ndarray] = {}
        total_weighted_cost_adv = np.zeros_like(norm_rew_adv, dtype=np.float32)

        for k in self.cost_keys:
            c_arr = np.asarray(raw_cost_advs.get(k, np.zeros_like(r_arr)), dtype=np.float32)
            if not np.all(np.isfinite(c_arr)):
                raise ValueError(f"Non-finite values in raw_cost_advs[{k}].")

            c_centered = c_arr - np.mean(c_arr)
            c_std = float(np.std(c_arr))
            if len(c_arr) > 1 and c_std > 1e-6:
                norm_c = c_centered / (c_std + 1e-8)
            else:
                norm_c = np.zeros_like(c_centered, dtype=np.float32)

            norm_cost_advs[k] = norm_c
            term = float(self.lambdas[k]) * norm_c
            weighted_cost_terms[k] = term
            total_weighted_cost_adv = total_weighted_cost_adv + term

        if episode_weights is not None:
            w_arr = np.asarray(episode_weights, dtype=np.float32)
            if len(w_arr) != len(norm_rew_adv):
                raise ValueError("episode_weights length mismatch.")
            total_weighted_cost_adv = total_weighted_cost_adv * w_arr

        lagrangian_adv = norm_rew_adv - total_weighted_cost_adv

        if not np.all(np.isfinite(lagrangian_adv)):
            raise ValueError("Non-finite values in computed lagrangian_adv.")

        diag = {
            "norm_rew_adv": norm_rew_adv,
            "norm_cost_advs": norm_cost_advs,
            "weighted_cost_terms": weighted_cost_terms,
            "total_weighted_cost_adv": total_weighted_cost_adv,
            "mean_norm_rew_adv": float(np.mean(norm_rew_adv)),
            "std_norm_rew_adv": float(np.std(norm_rew_adv)),
            "mean_weighted_cost_adv": float(np.mean(total_weighted_cost_adv)),
            "std_weighted_cost_adv": float(np.std(total_weighted_cost_adv)),
            "mean_lagrangian_adv": float(np.mean(lagrangian_adv)),
            "std_lagrangian_adv": float(np.std(lagrangian_adv)),
        }
        return lagrangian_adv.astype(np.float32), diag


class ConstrainedMAPPOTrainerV2:
    """
    Remediated Constrained MAPPO Trainer with:
      - Rollout batches across multiple episodes
      - Proper GAE + multi-epoch minibatch PPO
      - Corrected pre-multiplier cost advantage normalization
      - Optional Episodic Rockafellar-Uryasev Tail-Risk Weighting
      - Fixed validation suite checkpoint selection
    """

    def __init__(
        self,
        config: MAPPOConfig,
        constrained_cfg_path: Optional[str] = None,
        econ_cfg_path: Optional[str] = None,
        adaptive_multipliers: bool = True,
        use_tail_risk: bool = False,
        enable_adequacy_fallback_during_train: bool = False,
    ):
        self.config = config
        self.cfg_path = str(constrained_cfg_path or DEFAULT_CONSTRAINED_V2_YAML)
        with open(self.cfg_path, "r", encoding="utf-8") as f:
            self.v2_cfg = yaml.safe_load(f)

        self.ppo_cfg = self.v2_cfg["ppo_update"]
        self.tail_cfg = self.v2_cfg["tail_risk_cvar"]
        self.adaptive_multipliers = adaptive_multipliers
        self.use_tail_risk = use_tail_risk
        self.enable_adequacy_fallback_during_train = enable_adequacy_fallback_during_train

        self.econ_cfg = EconomicConfig.from_yaml(econ_cfg_path)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.log_dir = Path(config.log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Training environment: enable adequacy checker (audit-only by default
        # so the policy learns to act rather than hiding behind fallback!)
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

        # Seed torch & numpy
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        obs_dims = {ag: self.env.observation_space(ag).shape[0] for ag in self.env.possible_agents}
        self.agent = MAPPOAgent(
            config=config,
            obs_dims=obs_dims,
            action_spaces=self.env._action_spaces,
            state_dim=self.env._global_state_space.shape[0],
        )

        # Relax generator initial negative bias prior (-2.5 -> -0.6) so exploration
        # can actually reach the >0.04 MW startup threshold during outages
        if "generator" in self.agent.actors and hasattr(self.agent.actors["generator"], "mean_head"):
            nn.init.constant_(self.agent.actors["generator"].mean_head.bias, -0.6)

        # Override optimizer LRs with calibrated v2 PPO values
        for ag, opt in self.agent.actor_optimizers.items():
            for pg in opt.param_groups:
                pg["lr"] = float(self.ppo_cfg["actor_lr"])
        for pg in self.agent.critic_optimizer.param_groups:
            pg["lr"] = float(self.ppo_cfg["critic_lr"])

        self.checkpoint_manager = CheckpointManager(checkpoint_dir=str(self.checkpoint_dir))
        self.cost_evaluator = SafetyConstraintEvaluatorV2(self.cfg_path)
        self.lagrange_mgr = AdaptiveLagrangeManagerV2(self.cfg_path, adaptive=adaptive_multipliers)

        self.training_history: List[Dict[str, Any]] = []
        self.multiplier_history: List[Dict[str, Any]] = []
        self.validation_history: List[Dict[str, Any]] = []
        self.best_val_score = -float("inf")
        self.best_checkpoint_episode = -1

    def _compute_tail_weights(self, ep_social_costs: List[float], ep_lengths: List[int]) -> np.ndarray:
        """
        Variational Rockafellar-Uryasev Episodic Tail-Weighting over rollout batch.
        Computes empirical VaR_alpha across episode true social costs Z_e and
        weights tail episodes proportionally to excess (Z_e - VaR)+.
        """
        n_eps = len(ep_social_costs)
        if not self.use_tail_risk or n_eps < 2:
            return np.ones(sum(ep_lengths), dtype=np.float32)

        alpha = float(self.tail_cfg["alpha"])
        beta = float(self.tail_cfg["tail_weight_beta"])
        z = np.asarray(ep_social_costs, dtype=np.float32)
        var_nu = float(np.percentile(z, 100.0 * (1.0 - alpha)))
        z_span = float(max(1.0, np.max(z) - np.min(z)))

        step_weights = []
        for cost_e, length_e in zip(z, ep_lengths):
            excess = max(0.0, float(cost_e) - var_nu)
            w_e = 1.0 + beta * (excess / (alpha * z_span))
            w_e = float(np.clip(w_e, 1.0, 1.0 + beta / max(alpha, 0.05)))
            step_weights.extend([w_e] * length_e)
        return np.asarray(step_weights, dtype=np.float32)

    def train(self, total_episodes: int = 80, val_interval_episodes: int = 8) -> Dict[str, Any]:
        """
        Execute corrected Constrained MAPPO training with fixed validation selection.
        """
        episodes_per_update = int(self.ppo_cfg["episodes_per_update"])
        ppo_epochs = int(self.ppo_cfg["ppo_epochs"])
        minibatch_size = int(self.ppo_cfg["minibatch_size"])
        gamma = float(self.ppo_cfg["gamma"])
        gae_lambda = float(self.ppo_cfg["gae_lambda"])
        clip_eps = float(self.ppo_cfg["clip_eps"])
        entropy_coef = float(self.ppo_cfg["entropy_coef"])
        max_grad_norm = float(self.ppo_cfg["max_grad_norm"])

        print(
            f"[*] Starting Phase 4.4.2 Constrained MAPPO v2 | Seed={self.config.seed} "
            f"| Episodes={total_episodes} | TailRisk={self.use_tail_risk}",
            flush=True,
        )

        # Initial baseline validation at Ep 0 (intrinsic policy performance without fallback masking)
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
        print(
            f"  [Val Ep 000] Score={init_val['val_score']:+.1f} | "
            f"SocialCost=${init_val['val_true_social_cost_usd']:.1f} | "
            f"CritRel={init_val['val_critical_reliability_pct']:.2f}% | "
            f"Inact={init_val['val_inaction_rate_pct']:.1f}%",
            flush=True,
        )

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
            batch_ep_mean_costs: Dict[str, List[float]] = {k: [] for k in self.lagrange_mgr.cost_keys}
            batch_ep_social_costs: List[float] = []
            batch_ep_lengths: List[int] = []

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

                inaction_steps = 0
                emerg_steps = 0
                boundary_steps = 0
                feasible_actions = 0
                total_actions = 0
                ep_reward = 0.0
                done = False

                while self.env.agents and not done:
                    proposed_actions, log_probs = self.agent.select_actions(obs, deterministic=False)
                    val = float(self.agent.get_value(global_state))

                    next_obs, rewards, terms, truncs, next_infos = self.env.step(proposed_actions)
                    last_st = self.env.episode_history[-1]
                    diag = next_infos.get("battery", {}).get("shield_diagnostics", {})
                    adeq = next_infos.get("battery", {}).get("adequacy_diagnostics", {})

                    for ag, st_cls in diag.get("action_classification", {}).items():
                        total_actions += 1
                        if st_cls == "FEASIBLE":
                            feasible_actions += 1
                    if diag.get("any_emergency", False):
                        emerg_steps += 1
                    if diag.get("any_boundary", False):
                        boundary_steps += 1
                    if adeq.get("inaction_detected", False):
                        inaction_steps += 1

                    step_costs = self.cost_evaluator.evaluate_step_costs(
                        state_dict=last_st,
                        raw_actions=proposed_actions,
                        adequacy_diagnostics=adeq,
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
                    for k in self.lagrange_mgr.cost_keys:
                        ep_costs[k].append(step_costs[k])

                    obs = next_obs
                    global_state = next_infos.get("battery", {}).get("global_state", global_state)
                    done = done_step

                # Episode GAE for rewards
                T_ep = len(ep_rewards)
                last_val = 0.0 if any(terms.values()) else float(self.agent.get_value(global_state))
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

                # Discounted cost returns per constraint (centered baseline within episode/batch)
                for k in self.lagrange_mgr.cost_keys:
                    c_seq = ep_costs[k]
                    c_ret = np.zeros(T_ep, dtype=np.float32)
                    run_c = 0.0
                    for t_i in reversed(range(T_ep)):
                        run_c = c_seq[t_i] + gamma * run_c
                        c_ret[t_i] = run_c
                    batch_raw_cost_advs[k].append(c_ret)
                    batch_ep_mean_costs[k].append(float(np.mean(c_seq)))

                econ = compute_episode_economics(self.env.episode_history, dt_hours=self.env.dt_hours, config=self.econ_cfg)
                batch_ep_social_costs.append(econ["true_social_cost_usd"])
                batch_ep_lengths.append(T_ep)

                ep_rec = {
                    "episode": ep_idx,
                    "reward": round(ep_reward, 2),
                    "operating_cost_usd": econ["operating_cost_usd"],
                    "true_social_cost_usd": econ["true_social_cost_usd"],
                    "critical_ens_mwh": econ["critical_ens_mwh"],
                    "critical_reliability_pct": econ["critical_reliability_pct"],
                    "feasible_pct": round((feasible_actions / total_actions * 100.0) if total_actions else 100.0, 2),
                    "inaction_steps": inaction_steps,
                    "emergency_steps": emerg_steps,
                    "boundary_steps": boundary_steps,
                    "duration_s": round(time.time() - ep_start, 2),
                }
                self.training_history.append(ep_rec)

            # ------------------------------------------------------------
            # Batch PPO Update with Corrected Pre-Normalized Lagrangian Adv
            # ------------------------------------------------------------
            update_idx += 1
            flat_returns = np.concatenate(batch_returns, axis=0)
            flat_raw_rew_adv = np.concatenate(batch_raw_rew_adv, axis=0)
            flat_raw_cost_advs = {k: np.concatenate(batch_raw_cost_advs[k], axis=0) for k in self.lagrange_mgr.cost_keys}
            tail_weights = self._compute_tail_weights(batch_ep_social_costs, batch_ep_lengths)

            lagrangian_adv, adv_diag = self.lagrange_mgr.compute_lagrangian_advantages(
                raw_reward_adv=flat_raw_rew_adv,
                raw_cost_advs=flat_raw_cost_advs,
                episode_weights=tail_weights if self.use_tail_risk else None,
            )

            # Multi-epoch minibatch PPO updates
            N_batch = len(lagrangian_adv)
            flat_states = np.asarray(batch_states, dtype=np.float32)

            actor_tensors = {}
            for ag in self.env.possible_agents:
                act_arr = np.asarray(batch_acts[ag])
                act_dtype = torch.long if ag in ("load_management", "grid_resilience") else torch.float32
                actor_tensors[ag] = {
                    "obs": torch.tensor(np.asarray(batch_obs[ag], dtype=np.float32), dtype=torch.float32),
                    "acts": torch.tensor(act_arr, dtype=act_dtype),
                    "old_lp": torch.tensor(np.asarray(batch_log_probs[ag], dtype=np.float32), dtype=torch.float32),
                }
            adv_tensor = torch.tensor(lagrangian_adv, dtype=torch.float32)
            ret_tensor = torch.tensor(flat_returns, dtype=torch.float32).unsqueeze(-1)
            state_tensor = torch.tensor(flat_states, dtype=torch.float32)

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
                        mb_adv = adv_tensor[mb_idx]
                        surr1 = ratio * mb_adv
                        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                        p_loss = -torch.min(surr1, surr2).mean()
                        e_loss = -entropy_coef * entropy.mean()
                        loss = p_loss + e_loss

                        if not torch.isfinite(loss):
                            raise RuntimeError(f"Non-finite actor loss for {ag}: {loss.item()}")
                        opt.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                        opt.step()

                    # Critic update
                    pred_v = self.agent.critic(state_tensor[mb_idx])
                    v_loss = nn.functional.mse_loss(pred_v, ret_tensor[mb_idx])
                    if not torch.isfinite(v_loss):
                        raise RuntimeError(f"Non-finite critic loss: {v_loss.item()}")
                    self.agent.critic_optimizer.zero_grad()
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.critic.parameters(), max_grad_norm)
                    self.agent.critic_optimizer.step()
                    last_v_loss = float(v_loss.item())

            # Dual multiplier update on batch mean constraint costs
            mean_batch_costs = {k: float(np.mean(batch_ep_mean_costs[k])) for k in self.lagrange_mgr.cost_keys}
            new_lams = self.lagrange_mgr.update_multipliers(mean_batch_costs)

            mult_rec = {
                "update": update_idx,
                "episode": ep_idx,
                "critic_loss": round(last_v_loss, 4),
                "mean_norm_rew_adv": round(adv_diag["mean_norm_rew_adv"], 4),
                "std_norm_rew_adv": round(adv_diag["std_norm_rew_adv"], 4),
                "mean_weighted_cost_adv": round(adv_diag["mean_weighted_cost_adv"], 4),
                "std_weighted_cost_adv": round(adv_diag["std_weighted_cost_adv"], 4),
                "mean_lagrangian_adv": round(adv_diag["mean_lagrangian_adv"], 4),
                "std_lagrangian_adv": round(adv_diag["std_lagrangian_adv"], 4),
                **{f"mean_cost_{k}": round(mean_batch_costs[k], 4) for k in self.lagrange_mgr.cost_keys},
                **{f"mean_cost_adv_{k}": round(float(np.mean(adv_diag["norm_cost_advs"][k])), 4) for k in self.lagrange_mgr.cost_keys},
                **{k: round(v, 4) for k, v in new_lams.items()},
            }
            self.multiplier_history.append(mult_rec)

            # Fixed Validation Suite Evaluation & Checkpoint Selection
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
                self.validation_history.append(val_res)

                self.checkpoint_manager.save_checkpoint(
                    agent=self.agent,
                    episode=ep_idx,
                    metrics=val_res,
                    is_best=is_best,
                )
                print(
                    f"  [Val Ep {ep_idx:03d}] Score={val_res['val_score']:+.1f} | "
                    f"SocialCost=${val_res['val_true_social_cost_usd']:.1f} | "
                    f"CritRel={val_res['val_critical_reliability_pct']:.2f}% | "
                    f"Inact={val_res['val_inaction_rate_pct']:.1f}% | "
                    f"Best={'YES' if is_best else 'no'} | lam_crit={new_lams['lambda_crit']:.2f}",
                    flush=True,
                )

        # Persist logs
        pd.DataFrame(self.training_history).to_csv(self.log_dir / "training_episodes.csv", index=False)
        pd.DataFrame(self.multiplier_history).to_csv(self.log_dir / "multiplier_advantages_history.csv", index=False)
        pd.DataFrame(self.validation_history).to_csv(self.log_dir / "validation_selection_history.csv", index=False)

        return {
            "best_checkpoint_episode": self.best_checkpoint_episode,
            "best_val_score": self.best_val_score,
            "total_episodes": ep_idx,
            "total_steps": ep_idx * self.config.max_steps,
            "final_multipliers": self.lagrange_mgr.get_multipliers(),
        }
