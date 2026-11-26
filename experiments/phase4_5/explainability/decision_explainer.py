"""
experiments/phase4_5/explainability/decision_explainer.py
=========================================================
GRIDGUARD AI — Phase 4.5 Agent-Level Decision Explanations, Reward
Decomposition Exposure, and Safety/Fallback Intervention Tracing.

Generates grounded, verifiable explanations for all 5 decentralized agents
directly from actual environment telemetry, action decoding, SafetyShield
diagnostics, OperationalAdequacyChecker diagnostics, and reward breakdown.
Never fabricates physical justifications that contradict simulation state.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch

from rl_environment.agent_config import AGENTS, CAPACITY_SCALING
from rl_environment.action_space import parse_agent_action, GRID_ACTIONS, LOAD_ACTIONS
from experiments.phase4_5.explainability.feature_attribution import get_agent_feature_names


REWARD_COMPONENT_KEYS = [
    "r_cost",
    "r_crit_ens",
    "r_flex_ens",
    "r_other_ens",
    "r_emissions",
    "r_degrad",
    "r_voltage",
    "r_imbalance",
    "r_curt",
    "r_reserve",
    "r_service",
]


def verify_checkpoint_metadata(
    checkpoint_path: Path,
    expected_experiment: Optional[str] = "bat_only",
    expected_seed: Optional[int] = 202,
    expected_episode: Optional[int] = 80,
) -> Dict[str, Any]:
    """Verify candidate checkpoint file exists and matches expected training provenance."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ep = int(ckpt.get("episode", -1))
    metrics = ckpt.get("metrics", {}) or {}
    exp = metrics.get("experiment")
    seed = metrics.get("seed")

    if expected_experiment is not None and exp != expected_experiment:
        raise ValueError(f"Checkpoint experiment mismatch: expected {expected_experiment}, got {exp}")
    if expected_seed is not None and int(seed) != int(expected_seed):
        raise ValueError(f"Checkpoint seed mismatch: expected {expected_seed}, got {seed}")
    if expected_episode is not None and ep != int(expected_episode):
        raise ValueError(f"Checkpoint episode mismatch: expected {expected_episode}, got {ep}")

    actors = ckpt.get("actors", {})
    missing_agents = [ag for ag in AGENTS if ag not in actors]
    if missing_agents:
        raise ValueError(f"Checkpoint missing actor weights for agents: {missing_agents}")

    return {
        "checkpoint_path": str(ckpt_path),
        "episode": ep,
        "experiment": exp,
        "seed": seed,
        "metrics": metrics,
        "agents_present": list(actors.keys()),
        "verified": True,
    }


def _format_hour_timestamp(hour: float) -> str:
    total_min = int(round((float(hour) % 24.0) * 60.0))
    hh = (total_min // 60) % 24
    mm = total_min % 60
    return f"{hh:02d}:{mm:02d}"


def _extract_relevant_observations(agent: str, obs_vec: Optional[np.ndarray]) -> Dict[str, float]:
    """Map raw numpy observation vector cleanly to named features (safe on missing telemetry)."""
    if obs_vec is None:
        return {}
    names = get_agent_feature_names(agent, forecast_mode=len(obs_vec) > 13)
    arr = np.asarray(obs_vec, dtype=np.float32).flatten()
    out: Dict[str, float] = {}
    for i, val in enumerate(arr):
        fname = names[i] if i < len(names) else f"obs_{i}"
        v_clean = 0.0 if not np.isfinite(val) else float(val)
        out[fname] = round(v_clean, 4)
    return out


def _format_raw_and_decoded(agent: str, raw_val: Any, exec_val: Any) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """Safely parse raw and executed actions using environment's exact decoder."""
    try:
        dec_raw = parse_agent_action(agent, raw_val)
    except Exception:
        dec_raw = {"error": f"invalid_raw_action:{raw_val}"}
    try:
        dec_exec = parse_agent_action(agent, exec_val)
    except Exception:
        dec_exec = {"error": f"invalid_exec_action:{exec_val}"}

    if isinstance(raw_val, np.ndarray):
        raw_out = [round(float(x), 4) for x in raw_val.flatten()]
    elif np.isscalar(raw_val):
        raw_out = int(raw_val) if isinstance(raw_val, (int, np.integer)) else round(float(raw_val), 4)
    else:
        raw_out = raw_val
    return raw_out, dec_raw, dec_exec


class DecisionExplainer:
    """
    Constructs step-by-step agent decision logs, safety/fallback intervention
    traces, and reward component decompositions from real simulation telemetry.
    """

    def __init__(self, controller_name: str, mode: str):
        self.controller_name = controller_name
        self.mode = mode
        self.agent_records: List[Dict[str, Any]] = []
        self.intervention_records: List[Dict[str, Any]] = []
        self.reward_records: List[Dict[str, Any]] = []

    def reset(self):
        self.agent_records.clear()
        self.intervention_records.clear()
        self.reward_records.clear()

    def record_step(
        self,
        scenario_id: str,
        eval_seed: int,
        step_idx: int,
        pre_state: Dict[str, Any],
        obs_dict: Dict[str, np.ndarray],
        raw_actions: Dict[str, Any],
        post_state: Dict[str, Any],
        shield_diag: Dict[str, Any],
        adeq_diag: Dict[str, Any],
        reward_breakdown: Dict[str, float],
        shared_reward: float,
    ) -> None:
        hour = float(pre_state.get("hour", post_state.get("hour", 0.0)))
        ts = _format_hour_timestamp(hour)

        shield_exec = shield_diag.get("executed_actions", raw_actions) or raw_actions
        shield_interventions = shield_diag.get("interventions", {}) or {}
        shield_class = shield_diag.get("action_classification", {}) or {}
        shield_viols = list(shield_diag.get("violations", []) or [])
        shield_reasons = list(shield_diag.get("reasons", []) or [])

        adeq_intervened = bool(adeq_diag.get("adequacy_intervention", False))
        adeq_reasons = list(adeq_diag.get("intervention_reasons", []) or [])

        # Reconstruct final executed actions cleanly after both shield & fallback (defensive against None/malformed)
        ren_fallback = shield_exec.get("renewable", [0.0, 0.0])
        try:
            ren_arr = np.asarray(ren_fallback, dtype=np.float32).flatten()
            if len(ren_arr) < 2 or not np.all(np.isfinite(ren_arr[:2])):
                ren_arr = np.array([0.0, 0.0], dtype=np.float32)
        except Exception:
            ren_arr = np.array([0.0, 0.0], dtype=np.float32)

        def _safe_int(val: Any, default: int = 0) -> int:
            try:
                return int(val)
            except Exception:
                return default

        final_exec = {
            "renewable": np.array([
                float(1.0 - (post_state.get("solar_mw", 0.0) / max(post_state.get("solar_available_mw", 1e-6), 1e-6)))
                if float(post_state.get("solar_available_mw", 0.0)) > 1e-4 else float(ren_arr[0]),
                float(1.0 - (post_state.get("wind_mw", 0.0) / max(post_state.get("wind_available_mw", 1e-6), 1e-6)))
                if float(post_state.get("wind_available_mw", 0.0)) > 1e-4 else float(ren_arr[1]),
            ], dtype=np.float32),
            "battery": np.array([
                float(np.clip(float(post_state.get("battery_mw", 0.0)) / CAPACITY_SCALING["battery_power_mw"], -1.0, 1.0))
            ], dtype=np.float32),
            "generator": np.array([
                float(np.clip(float(post_state.get("generator_mw", 0.0)) / CAPACITY_SCALING["generator_capacity_mw"], 0.0, 1.0))
            ], dtype=np.float32),
            "load_management": _safe_int(shield_exec.get("load_management", raw_actions.get("load_management", 0))),
            "grid_resilience": _safe_int(shield_exec.get("grid_resilience", raw_actions.get("grid_resilience", 0))),
        }
        # If fallback relaxed load shedding, check actual served flexible load
        if adeq_intervened and any("PREVENTED_UNNEEDED_SHED" in r for r in adeq_reasons):
            final_exec["load_management"] = 0

        # Reward decomposition verification
        comp_sum = sum(float(reward_breakdown.get(k, 0.0)) for k in REWARD_COMPONENT_KEYS)
        rew_rec = {
            "controller": self.controller_name,
            "mode": self.mode,
            "scenario": scenario_id,
            "eval_seed": int(eval_seed),
            "step": int(step_idx),
            "hour": round(hour, 2),
            "timestamp": ts,
            "reward_total": round(float(shared_reward), 6),
            "component_sum": round(comp_sum, 6),
            "decomposition_error": round(abs(comp_sum - float(shared_reward)), 8),
            **{k: round(float(reward_breakdown.get(k, 0.0)), 6) for k in REWARD_COMPONENT_KEYS},
            "net_step_cost_dollars": round(float(reward_breakdown.get("net_step_cost_dollars", 0.0)), 4),
            "crit_shed_mwh": round(float(reward_breakdown.get("crit_shed_mwh", 0.0)), 6),
            "flex_shed_mwh": round(float(reward_breakdown.get("flex_shed_mwh", 0.0)), 6),
        }
        self.reward_records.append(rew_rec)

        # Agent-level explanation records
        for ag in AGENTS:
            raw_v = raw_actions.get(ag)
            sh_v = shield_exec.get(ag, raw_v)
            ex_v = final_exec.get(ag, sh_v)
            raw_ser, dec_raw, dec_exec = _format_raw_and_decoded(ag, raw_v, ex_v)
            _, dec_sh, _ = _format_raw_and_decoded(ag, raw_v, sh_v)

            sh_mod = bool(shield_interventions.get(ag, False))
            fb_mod = self._did_fallback_modify_agent(ag, dec_sh, dec_exec, adeq_intervened, adeq_reasons, post_state)
            constraints = self._get_agent_constraints(ag, pre_state, post_state)
            explanation, trig_cond, corr_reason, op_effect, interv_stage = self._explain_agent_step(
                agent=ag,
                pre_state=pre_state,
                post_state=post_state,
                dec_raw=dec_raw,
                dec_sh=dec_sh,
                dec_exec=dec_exec,
                sh_mod=sh_mod,
                sh_cls=shield_class.get(ag, "FEASIBLE"),
                sh_viols=shield_viols,
                sh_reasons=shield_reasons,
                fb_mod=fb_mod,
                adeq_reasons=adeq_reasons,
                adeq_diag=adeq_diag,
            )

            rel_obs = _extract_relevant_observations(ag, obs_dict.get(ag))
            ag_rew_contrib = self._agent_reward_attribution(ag, reward_breakdown)

            rec = {
                "controller": self.controller_name,
                "mode": self.mode,
                "scenario": scenario_id,
                "eval_seed": int(eval_seed),
                "step": int(step_idx),
                "hour": round(hour, 2),
                "timestamp": ts,
                "agent": ag,
                "relevant_observations": rel_obs,
                "raw_policy_output": raw_ser,
                "decoded_action": dec_raw,
                "shielded_action": dec_sh,
                "executed_action": dec_exec,
                "shield_modified": sh_mod,
                "fallback_modified": fb_mod,
                "intervention_stage": interv_stage,
                "operational_constraints": constraints,
                "reward_total": round(float(shared_reward), 4),
                "agent_reward_components": ag_rew_contrib,
                "explanation": explanation,
            }
            self.agent_records.append(rec)

            if sh_mod or fb_mod:
                self.intervention_records.append({
                    "controller": self.controller_name,
                    "mode": self.mode,
                    "scenario": scenario_id,
                    "eval_seed": int(eval_seed),
                    "step": int(step_idx),
                    "hour": round(hour, 2),
                    "timestamp": ts,
                    "agent": ag,
                    "intervention_stage": interv_stage,
                    "triggering_condition": trig_cond,
                    "original_proposed_action": dec_raw,
                    "corrected_action": dec_exec,
                    "reason": corr_reason,
                    "prevented_violation": bool(sh_mod or bool(adeq_diag.get("inaction_detected", False))),
                    "operational_effect": op_effect,
                    "shield_violations": "; ".join(shield_viols) if sh_mod else "",
                    "adequacy_reasons": "; ".join(adeq_reasons) if fb_mod else "",
                })

    @staticmethod
    def _did_fallback_modify_agent(
        agent: str,
        dec_sh: Dict[str, Any],
        dec_exec: Dict[str, Any],
        adeq_intervened: bool,
        adeq_reasons: List[str],
        post_state: Dict[str, Any],
    ) -> bool:
        if not adeq_intervened:
            return False
        if agent == "battery" and any("ADEQUACY_BATTERY_DISPATCH" in r for r in adeq_reasons):
            return True
        if agent == "generator" and any("ADEQUACY_GENERATOR" in r for r in adeq_reasons):
            return True
        if agent == "load_management" and any("ADEQUACY_PREVENTED_UNNEEDED_SHED" in r for r in adeq_reasons):
            return True
        return False

    @staticmethod
    def _get_agent_constraints(agent: str, pre_state: Dict[str, Any], post_state: Dict[str, Any]) -> Dict[str, Any]:
        soc_pre = float(pre_state.get("battery_soc", 80.0))
        soc_post = float(post_state.get("battery_soc", 80.0))
        if agent == "battery":
            return {
                "soc_pre_pct": round(soc_pre, 2),
                "soc_post_pct": round(soc_post, 2),
                "min_soc_pct": 20.0,
                "emergency_reserve_soc_pct": 30.0,
                "max_soc_pct": 95.0,
                "power_limits_mw": [-0.25, 0.25],
            }
        elif agent == "generator":
            return {
                "gen_state_pre": pre_state.get("generator_state", "OFF"),
                "gen_state_post": post_state.get("generator_state", "OFF"),
                "capacity_mw": 0.40,
                "min_stable_mw": 0.08,
                "grid_connected": bool(post_state.get("grid_connected", True)),
            }
        elif agent == "load_management":
            return {
                "critical_mw": round(float(post_state.get("nominal_critical_mw", 0.0)), 4),
                "flexible_mw": round(float(post_state.get("nominal_flexible_mw", 0.0)), 4),
                "other_mw": round(float(post_state.get("nominal_other_mw", 0.0)), 4),
                "critical_protected": True,
            }
        elif agent == "grid_resilience":
            return {
                "grid_available": bool(post_state.get("grid_available", True)),
                "sync_verified": bool(post_state.get("sync_verified", True)),
                "voltage_bounds_pu": [0.95, 1.05],
                "pcc_voltage_pu": round(float(post_state.get("pcc_voltage_pu", 1.0)), 4),
            }
        return {
            "solar_avail_mw": round(float(post_state.get("solar_available_mw", 0.0)), 4),
            "wind_avail_mw": round(float(post_state.get("wind_available_mw", 0.0)), 4),
        }

    @staticmethod
    def _agent_reward_attribution(agent: str, rb: Dict[str, float]) -> Dict[str, float]:
        if agent == "renewable":
            return {"r_curt": round(float(rb.get("r_curt", 0.0)), 4), "r_cost": round(float(rb.get("r_cost", 0.0)), 4)}
        if agent == "battery":
            return {
                "r_degrad": round(float(rb.get("r_degrad", 0.0)), 4),
                "r_reserve": round(float(rb.get("r_reserve", 0.0)), 4),
                "r_cost": round(float(rb.get("r_cost", 0.0)), 4),
                "r_crit_ens": round(float(rb.get("r_crit_ens", 0.0)), 4),
            }
        if agent == "generator":
            return {
                "r_cost": round(float(rb.get("r_cost", 0.0)), 4),
                "r_emissions": round(float(rb.get("r_emissions", 0.0)), 4),
                "r_crit_ens": round(float(rb.get("r_crit_ens", 0.0)), 4),
            }
        if agent == "load_management":
            return {
                "r_crit_ens": round(float(rb.get("r_crit_ens", 0.0)), 4),
                "r_flex_ens": round(float(rb.get("r_flex_ens", 0.0)), 4),
                "r_other_ens": round(float(rb.get("r_other_ens", 0.0)), 4),
                "r_service": round(float(rb.get("r_service", 0.0)), 4),
            }
        return {
            "r_voltage": round(float(rb.get("r_voltage", 0.0)), 4),
            "r_imbalance": round(float(rb.get("r_imbalance", 0.0)), 4),
            "r_cost": round(float(rb.get("r_cost", 0.0)), 4),
        }

    @staticmethod
    def _explain_agent_step(
        agent: str,
        pre_state: Dict[str, Any],
        post_state: Dict[str, Any],
        dec_raw: Dict[str, Any],
        dec_sh: Dict[str, Any],
        dec_exec: Dict[str, Any],
        sh_mod: bool,
        sh_cls: str,
        sh_viols: List[str],
        sh_reasons: List[str],
        fb_mod: bool,
        adeq_reasons: List[str],
        adeq_diag: Dict[str, Any],
    ) -> Tuple[str, str, str, str, str]:
        grid_conn = bool(post_state.get("grid_connected", True))
        grid_avail = bool(post_state.get("grid_available", True))
        re_p = float(post_state.get("re_gen_mw", 0.0))
        nom_tot = float(post_state.get("nominal_total_mw", 0.0))
        soc_pre = float(pre_state.get("battery_soc", 80.0))
        soc_post = float(post_state.get("battery_soc", 80.0))
        deficit_mw = max(0.0, nom_tot - re_p)

        if sh_mod and fb_mod:
            stage = "BOTH_SHIELD_AND_FALLBACK"
        elif fb_mod:
            stage = "ADEQUACY_FALLBACK"
        elif sh_mod:
            stage = "EMERGENCY_OVERRIDE" if sh_cls == "EMERGENCY_OVERRIDE" else "BOUNDARY_CLIP"
        else:
            stage = "NONE"

        trig_cond = "nominal"
        corr_reason = ""
        op_effect = "executed raw proposal directly"

        if agent == "renewable":
            s_curt = float(dec_raw.get("solar_curtailment_factor", 0.0))
            w_curt = float(dec_raw.get("wind_curtailment_factor", 0.0))
            ex_s = float(dec_exec.get("solar_curtailment_factor", s_curt))
            ex_w = float(dec_exec.get("wind_curtailment_factor", w_curt))
            if sh_mod:
                trig_cond = f"islanded deficit ({deficit_mw:.3f} MW) with proposed RE curtailment ({s_curt:.2f}/{w_curt:.2f})"
                corr_reason = "UNNECESSARY_RENEWABLE_CURTAILMENT_IN_DEFICIT: blocked curtailment during supply shortfall"
                op_effect = f"restored full RE output ({re_p:.3f} MW)"
                msg = f"Renewable agent proposed curtailment ({s_curt:.0%}/{w_curt:.0%}), overridden by SafetyShield to {ex_s:.0%}/{ex_w:.0%} during {deficit_mw:.3f} MW deficit."
            else:
                msg = f"Renewable agent dispatched {re_p:.3f} MW clean generation (curtailment {ex_s:.0%} solar, {ex_w:.0%} wind)."

        elif agent == "battery":
            raw_p = float(dec_raw.get("requested_power_mw", 0.0))
            exec_p = float(post_state.get("battery_mw", 0.0))
            if fb_mod:
                trig_cond = f"islanded deficit ({deficit_mw:.3f} MW) with unused battery headroom (SOC={soc_pre:.1f}%)"
                corr_reason = "; ".join(r for r in adeq_reasons if "BATTERY" in r)
                op_effect = f"battery output changed from {raw_p:+.3f} MW to {exec_p:+.3f} MW (SOC {soc_pre:.1f}%->{soc_post:.1f}%)"
                msg = f"Battery agent proposed {raw_p:+.3f} MW; OperationalAdequacyFallback intervened to discharge {exec_p:+.3f} MW to cover deficit (SOC {soc_pre:.1f}%->{soc_post:.1f}%)."
            elif sh_mod:
                trig_cond = f"SOC={soc_pre:.1f}% violated safety reserve/bound for command {raw_p:+.3f} MW"
                corr_reason = "; ".join(r for r in sh_reasons if "Battery" in r) or "battery SOC safety bound"
                op_effect = f"clamped battery command from {raw_p:+.3f} MW to {exec_p:+.3f} MW"
                msg = f"Battery agent proposed {raw_p:+.3f} MW at SOC={soc_pre:.1f}%, clipped by SafetyShield to {exec_p:+.3f} MW."
            elif exec_p < -0.01:
                msg = f"Battery agent discharged {abs(exec_p):.3f} MW (SOC {soc_pre:.1f}%->{soc_post:.1f}%) because net demand ({deficit_mw:.3f} MW) exceeded renewable supply while {'islanded' if not grid_conn else 'grid-connected'}."
            elif exec_p > 0.01:
                msg = f"Battery agent charged {exec_p:.3f} MW (SOC {soc_pre:.1f}%->{soc_post:.1f}%) with RE={re_p:.3f} MW vs demand={nom_tot:.3f} MW."
            else:
                msg = f"Battery agent remained idle ({exec_p:+.3f} MW, SOC={soc_post:.1f}%) while {'grid import covered net load' if grid_conn else 'preserving remaining reserve'}."

        elif agent == "generator":
            raw_p = float(dec_raw.get("requested_power_mw", 0.0))
            exec_p = float(post_state.get("generator_mw", 0.0))
            gen_st = str(post_state.get("generator_state", "OFF"))
            fuel_l = float(post_state.get("step_fuel_l", 0.0))
            if fb_mod:
                trig_cond = f"islanded critical/supply deficit ({deficit_mw:.3f} MW) with generator under-dispatched ({raw_p:.3f} MW)"
                corr_reason = "; ".join(r for r in adeq_reasons if "GENERATOR" in r)
                op_effect = f"generator adjusted to state={gen_st}, output={exec_p:.3f} MW (fuel={fuel_l:.2f} L)"
                msg = f"Generator agent proposed {raw_p:.3f} MW; OperationalAdequacyFallback commanded backup startup/dispatch (state={gen_st}, {exec_p:.3f} MW)."
            elif sh_mod:
                trig_cond = f"healthy grid-connected state (grid_conn={grid_conn}) with unnecessary gen request {raw_p:.3f} MW"
                corr_reason = "UNNECESSARY_GENERATOR_DISPATCH: utility grid healthy and normal"
                op_effect = f"blocked unnecessary startup/dispatch (actual={exec_p:.3f} MW, state={gen_st})"
                msg = f"Generator agent proposed {raw_p:.3f} MW on healthy grid; SafetyShield blocked unnecessary dispatch (state={gen_st}, {exec_p:.3f} MW)."
            elif exec_p > 0.01 or gen_st == "STARTING":
                msg = f"Generator agent operated in {gen_st} mode delivering {exec_p:.3f} MW (fuel {fuel_l:.2f} L) during {'outage/islanded deficit' if not grid_conn else 'cooldown/transition'}."
            else:
                msg = f"Generator agent remained OFF (0.000 MW) because {'utility grid supply was adequate' if grid_conn else 'demand/reserve conditions did not trigger start'}."

        elif agent == "load_management":
            act_name = dec_exec.get("command", dec_exec.get("action_name", str(dec_exec)))
            shed_crit = float(post_state.get("shed_critical_mw", 0.0))
            shed_flex = float(post_state.get("shed_flexible_mw", 0.0))
            shed_oth = float(post_state.get("shed_other_mw", 0.0))
            if fb_mod:
                trig_cond = "unnecessary load curtailment requested when fallback restored supply"
                corr_reason = "ADEQUACY_PREVENTED_UNNEEDED_SHED"
                op_effect = f"relaxed curtailment to {act_name}"
                raw_name = dec_raw.get("command", dec_raw.get("action_name", str(dec_raw)))
                msg = f"Load-management agent curtailed {raw_name}; fallback restored supply and relaxed shedding (actual shed: crit={shed_crit:.3f}, flex={shed_flex:.3f} MW)."
            elif sh_mod:
                trig_cond = "attempted critical load curtailment"
                corr_reason = "CRITICAL_LOAD_SHEDDING_REJECTED"
                op_effect = "protected critical bus"
                msg = f"Load-management agent critical shed attempt blocked by SafetyShield."
            elif shed_crit + shed_flex + shed_oth > 1e-4:
                msg = f"Load-management agent applied {act_name}; total shed={shed_crit+shed_flex+shed_oth:.3f} MW (crit={shed_crit:.3f}, flex={shed_flex:.3f}, other={shed_oth:.3f} MW) under {'islanded deficit' if not grid_conn else 'grid-connected curtailment'}."
            else:
                msg = f"Load-management agent maintained full load service ({act_name}, 0.000 MW shed)."

        else:  # grid_resilience
            raw_cmd = dec_raw.get("command", "STAY_CONNECTED")
            exec_cmd = dec_exec.get("command", raw_cmd)
            if sh_mod:
                trig_cond = f"grid_avail={grid_avail}, sync_verified={post_state.get('sync_verified', True)} incompatible with {raw_cmd}"
                corr_reason = "; ".join(v for v in sh_viols if "GRID" in v or "SYNCHRONIZATION" in v) or "grid safety lock"
                op_effect = f"enforced {exec_cmd} (grid_connected={grid_conn})"
                msg = f"Grid-resilience agent proposed {raw_cmd}; SafetyShield enforced {exec_cmd} because {'utility line was unavailable' if not grid_avail else 'PCC synchronization was not verified'}."
            else:
                msg = f"Grid-resilience agent executed {exec_cmd} (grid_available={grid_avail}, grid_connected={grid_conn}, V_pcc={float(post_state.get('pcc_voltage_pu', 1.0)):.4f} pu)."

        return msg, trig_cond, corr_reason, op_effect, stage
