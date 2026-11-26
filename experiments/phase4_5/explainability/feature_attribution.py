"""
experiments/phase4_5/explainability/feature_attribution.py
==========================================================
GRIDGUARD AI — Phase 4.5 Policy Feature Attribution

Implements two complementary attribution methods across BOTH continuous
(BoundedSquashedGaussianPolicy) and discrete (CategoricalPolicy) actors:
  1. Integrated Gradients (IG): path integral from zero/neutral baseline to
     current observation vector wrt deterministic action output (continuous)
     or target-action probability / logit margin (discrete).
  2. Observation Perturbation Sensitivity: controlled perturbation (+delta /
     feature ablation) measuring direct local responsiveness of policy outputs.

Clearly distinguishes policy sensitivity (gradient/perturbation wrt obs) from
causal physical simulator influence. Includes controlled perturbation sanity tests.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch

from rl_environment.agent_config import AGENTS, LOCAL_OBS_SCHEMA, FORECAST_OBS_AUGMENTATIONS


def get_agent_feature_names(agent: str, forecast_mode: bool = True) -> List[str]:
    base = list(LOCAL_OBS_SCHEMA[agent])
    if forecast_mode:
        base.extend(FORECAST_OBS_AUGMENTATIONS.get(agent, []))
    return base


def _policy_scalar_target(actor: torch.nn.Module, obs_t: torch.Tensor, agent: str, target_action_idx: Optional[int] = None) -> torch.Tensor:
    """
    Smooth differentiable scalar target for actor attributions:
      - Continuous actors: deterministic squashed physical command in [low, high].
        For renewable (2D), sum of solar + wind curtailment factors.
      - Discrete actors: probability P(a = a* | obs) from Softmax(logits).
    """
    if hasattr(actor, "action_mid") and hasattr(actor, "action_half"):
        mean_u, _ = actor.forward(obs_t)
        squashed = torch.tanh(mean_u)
        scaled = actor.action_mid + actor.action_half * squashed
        return scaled.sum(dim=-1)
    else:
        logits = actor.forward(obs_t)
        probs = torch.softmax(logits, dim=-1)
        if target_action_idx is None:
            with torch.no_grad():
                target_action_idx = int(torch.argmax(logits, dim=-1).item())
        return probs[..., target_action_idx]


def compute_integrated_gradients(
    actor: torch.nn.Module,
    agent: str,
    obs_vec: np.ndarray,
    baseline_vec: Optional[np.ndarray] = None,
    n_steps: int = 20,
    target_action_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute Riemann-approximated Integrated Gradients over n_steps from baseline to obs_vec.
    """
    obs_arr = np.asarray(obs_vec, dtype=np.float32).flatten()
    if not np.all(np.isfinite(obs_arr)):
        raise ValueError(f"Non-finite observation values passed to Integrated Gradients for {agent}.")
    if baseline_vec is None:
        base_arr = np.zeros_like(obs_arr, dtype=np.float32)
    else:
        base_arr = np.asarray(baseline_vec, dtype=np.float32).flatten()

    actor.eval()
    obs_t = torch.tensor(obs_arr, dtype=torch.float32).unsqueeze(0)
    base_t = torch.tensor(base_arr, dtype=torch.float32).unsqueeze(0)

    if target_action_idx is None and not hasattr(actor, "action_mid"):
        with torch.no_grad():
            target_action_idx = int(torch.argmax(actor.forward(obs_t), dim=-1).item())

    alphas = torch.linspace(1.0 / n_steps, 1.0, steps=n_steps, dtype=torch.float32)
    total_grads = torch.zeros_like(obs_t)

    for alpha in alphas:
        interp = (base_t + alpha * (obs_t - base_t)).detach().requires_grad_(True)
        out = _policy_scalar_target(actor, interp, agent, target_action_idx=target_action_idx).sum()
        grad = torch.autograd.grad(out, interp, retain_graph=False)[0]
        total_grads = total_grads + grad.detach()

    avg_grads = total_grads / float(n_steps)
    ig_tensor = ((obs_t - base_t) * avg_grads).squeeze(0).detach().numpy()
    ig_clean = np.nan_to_num(ig_tensor, nan=0.0, posinf=0.0, neginf=0.0)

    with torch.no_grad():
        f_obs = float(_policy_scalar_target(actor, obs_t, agent, target_action_idx=target_action_idx).item())
        f_base = float(_policy_scalar_target(actor, base_t, agent, target_action_idx=target_action_idx).item())

    completeness_err = abs(float(np.sum(ig_clean)) - (f_obs - f_base))
    names = get_agent_feature_names(agent, forecast_mode=len(obs_arr) > 13)
    attributions = {names[i] if i < len(names) else f"obs_{i}": float(ig_clean[i]) for i in range(len(obs_arr))}

    return {
        "agent": agent,
        "method": "integrated_gradients",
        "f_obs": round(f_obs, 6),
        "f_baseline": round(f_base, 6),
        "ig_sum": round(float(np.sum(ig_clean)), 6),
        "completeness_error": round(completeness_err, 6),
        "attributions": attributions,
    }


def compute_perturbation_sensitivity(
    actor: torch.nn.Module,
    agent: str,
    obs_vec: np.ndarray,
    delta: float = 0.10,
    target_action_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute symmetric finite-difference / feature ablation sensitivity |f(x + d*e_j) - f(x - d*e_j)| / (2*d)
    plus raw directional response df = f(x + d*e_j) - f(x).
    Binary features (e.g. grid_status_binary, grid_available_binary, storm_warning)
    are flipped 0 <-> 1 for exact occlusion sensitivity.
    """
    obs_arr = np.asarray(obs_vec, dtype=np.float32).flatten()
    if not np.all(np.isfinite(obs_arr)):
        raise ValueError(f"Non-finite observation values passed to perturbation sensitivity for {agent}.")

    actor.eval()
    obs_t = torch.tensor(obs_arr, dtype=torch.float32).unsqueeze(0)
    if target_action_idx is None and not hasattr(actor, "action_mid"):
        with torch.no_grad():
            target_action_idx = int(torch.argmax(actor.forward(obs_t), dim=-1).item())

    with torch.no_grad():
        f_base = float(_policy_scalar_target(actor, obs_t, agent, target_action_idx=target_action_idx).item())

    names = get_agent_feature_names(agent, forecast_mode=len(obs_arr) > 13)
    sensitivities: Dict[str, float] = {}
    directional_deltas: Dict[str, float] = {}

    for i in range(len(obs_arr)):
        fname = names[i] if i < len(names) else f"obs_{i}"
        is_binary = ("binary" in fname) or ("onehot" in fname)
        x_p = obs_arr.copy()
        x_m = obs_arr.copy()
        if is_binary:
            flipped = 0.0 if obs_arr[i] >= 0.5 else 1.0
            x_p[i] = flipped
            with torch.no_grad():
                fp = float(_policy_scalar_target(actor, torch.tensor(x_p).unsqueeze(0), agent, target_action_idx=target_action_idx).item())
            sens = abs(fp - f_base)
            d_raw = fp - f_base
        else:
            x_p[i] = obs_arr[i] + delta
            x_m[i] = obs_arr[i] - delta
            with torch.no_grad():
                fp = float(_policy_scalar_target(actor, torch.tensor(x_p).unsqueeze(0), agent, target_action_idx=target_action_idx).item())
                fm = float(_policy_scalar_target(actor, torch.tensor(x_m).unsqueeze(0), agent, target_action_idx=target_action_idx).item())
            sens = abs(fp - fm) / (2.0 * delta)
            d_raw = fp - f_base

        sensitivities[fname] = float(np.nan_to_num(sens, nan=0.0))
        directional_deltas[fname] = float(np.nan_to_num(d_raw, nan=0.0))

    return {
        "agent": agent,
        "method": "perturbation_sensitivity",
        "f_obs": round(f_base, 6),
        "sensitivities": sensitivities,
        "directional_deltas": directional_deltas,
    }


def run_controlled_perturbation_validation(
    mappo_agent: Any,
    sample_obs_by_agent: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    """
    Verify feature attribution consistency via controlled semantic observation perturbations:
      1. Battery: Lowering battery_soc_norm & reserve_margin vs raising power_deficit_norm
      2. Generator: Switching grid_available_binary 1 -> 0 & raising power_deficit_norm
      3. Load Management: Raising power_deficit_norm during islanding
      4. Grid Resilience: Switching grid_available_binary 1 -> 0 & raising outage_risk_norm
      5. Renewable: Changing solar_available_norm & fc_solar_1h_norm
    """
    results: List[Dict[str, Any]] = []
    specs = [
        ("battery", "power_deficit_norm", +0.50, "Increase normalized deficit by +0.50"),
        ("battery", "emergency_reserve_margin", -0.35, "Drop reserve margin by -0.35"),
        ("battery", "grid_status_binary", -1.0, "Force grid status 1->0 (islanded)"),
        ("generator", "grid_available_binary", -1.0, "Force grid available 1->0"),
        ("generator", "power_deficit_norm", +0.50, "Increase normalized deficit by +0.50"),
        ("load_management", "power_deficit_norm", +0.50, "Increase normalized deficit by +0.50"),
        ("load_management", "grid_status_binary", -1.0, "Force grid status 1->0"),
        ("grid_resilience", "grid_available_binary", -1.0, "Force utility grid lost 1->0"),
        ("grid_resilience", "outage_risk_norm", +0.80, "Raise storm/outage risk +0.80"),
        ("renewable", "solar_available_norm", +0.40, "Raise solar availability +0.40"),
    ]

    for ag, feat, delta, desc in specs:
        if ag not in sample_obs_by_agent or ag not in mappo_agent.actors:
            continue
        obs = np.asarray(sample_obs_by_agent[ag], dtype=np.float32).copy()
        names = get_agent_feature_names(ag, forecast_mode=len(obs) > 13)
        if feat not in names:
            continue
        idx = names.index(feat)
        actor = mappo_agent.actors[ag]

        obs_pert = obs.copy()
        if "binary" in feat:
            obs_pert[idx] = 0.0 if obs[idx] >= 0.5 else 1.0
        else:
            obs_pert[idx] = float(np.clip(obs[idx] + delta, -2.0, 2.0))

        with torch.no_grad():
            f_orig = float(_policy_scalar_target(actor, torch.tensor(obs).unsqueeze(0), ag).item())
            f_pert = float(_policy_scalar_target(actor, torch.tensor(obs_pert).unsqueeze(0), ag).item())

        ig_res = compute_integrated_gradients(actor, ag, obs)
        results.append({
            "agent": ag,
            "feature": feat,
            "perturbation": desc,
            "orig_val": round(float(obs[idx]), 4),
            "pert_val": round(float(obs_pert[idx]), 4),
            "orig_policy_target": round(f_orig, 6),
            "pert_policy_target": round(f_pert, 6),
            "policy_delta": round(f_pert - f_orig, 6),
            "ig_attribution": round(float(ig_res["attributions"].get(feat, 0.0)), 6),
            "ig_completeness_error": ig_res["completeness_error"],
        })
    return results
