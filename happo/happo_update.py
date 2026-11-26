"""
happo/happo_update.py
=====================
Canonical Sequential HAPPO Optimization (Kuba et al., ICLR 2022):
  1. Start with cumulative importance ratio factor F^(1) = 1.
  2. Iterate sequentially over update_order (i_1, ..., i_n).
  3. For agent i_k:
       - Compute decomposed advantage M^(i_k) = F^(k).detach() * A_hat(s, a)
       - Optimize ONLY actor i_k over ppo_epochs minibatches using PPO clip + entropy
       - After completing agent i_k's update, evaluate post-update log pi_new^(i_k)
         (no_grad) and update F^(k+1) = F^(k) * exp(log pi_new - log pi_old)
  4. Update the centralized critic V_phi(s) over global_states and GAE returns.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn

from happo.happo_agent import HAPPOAgent
from happo.happo_buffer import HAPPORolloutBuffer
from happo.happo_advantage import (
    compute_cumulative_importance_factor,
    compute_decomposed_advantage,
)


def update_happo(
    agent: HAPPOAgent,
    buffer: HAPPORolloutBuffer,
    update_order: Optional[List[str]] = None,
    ppo_epochs: int = 6,
    minibatch_size: int = 128,
    clip_eps: float = 0.20,
    entropy_coef: float = 0.015,
    value_loss_coef: float = 0.50,
    max_grad_norm: float = 0.50,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Execute canonical sequential HAPPO actor updates followed by centralized critic optimization.
    """
    order = list(update_order or agent.config.update_order)
    if sorted(order) != sorted(agent.actors.keys()):
        raise ValueError(f"Invalid update_order {order}; expected permutation of {list(agent.actors.keys())}")

    tensors = buffer.get_full_tensors()
    num_steps = tensors["num_steps"]
    if num_steps == 0:
        raise ValueError("Cannot run update_happo on an empty HAPPORolloutBuffer.")

    obs = tensors["obs"]
    proposed_actions = tensors["proposed_actions"]
    old_log_probs = tensors["old_log_probs"]
    global_states = tensors["global_states"]
    old_values = tensors["old_values"]
    returns = tensors["returns"]
    joint_advantages = tensors["advantages"]

    # Initialize cumulative importance factor F^(1) = 1 across all N steps
    factor = torch.ones(num_steps, dtype=torch.float32, device=agent.device)

    actor_losses: Dict[str, List[float]] = {ag: [] for ag in order}
    actor_entropies: Dict[str, List[float]] = {ag: [] for ag in order}
    actor_kls: Dict[str, List[float]] = {ag: [] for ag in order}
    actor_clip_fracs: Dict[str, List[float]] = {ag: [] for ag in order}
    actor_grad_norms: Dict[str, List[float]] = {ag: [] for ag in order}
    factor_in_means: Dict[str, float] = {}
    factor_out_means: Dict[str, float] = {}
    post_ratio_means: Dict[str, float] = {}
    factor_stats_per_agent: Dict[str, Dict[str, float]] = {}

    # =====================================================================
    # 1. Sequential Actor Updates (Canonical Advantage Decomposition)
    # =====================================================================
    for ag in order:
        actor = agent.actors[ag]
        opt = agent.actor_optimizers[ag]
        ag_obs = obs[ag]
        ag_act = proposed_actions[ag]
        ag_old_lp = old_log_probs[ag]

        # Fixed prefix factor for agent i_k during its PPO epochs (detached from prior actors)
        factor_k = factor.detach().clone()
        factor_in_means[ag] = float(factor_k.mean().item())
        decomposed_adv = compute_decomposed_advantage(joint_advantages, factor_k)

        for _ in range(ppo_epochs):
            for mb_np in HAPPORolloutBuffer.iter_minibatch_indices(num_steps, minibatch_size, rng=rng):
                mb_idx = torch.as_tensor(mb_np, dtype=torch.long, device=agent.device)

                new_log_prob, entropy = actor.evaluate_actions(ag_obs[mb_idx], ag_act[mb_idx])
                ratio = torch.exp(new_log_prob - ag_old_lp[mb_idx])

                mb_adv = decomposed_adv[mb_idx]
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy_coef * entropy.mean()
                total_actor_loss = policy_loss + entropy_loss

                opt.zero_grad()
                total_actor_loss.backward()
                gn = nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                opt.step()

                with torch.no_grad():
                    approx_kl = (ag_old_lp[mb_idx] - new_log_prob).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()
                    actor_losses[ag].append(float(policy_loss.item()))
                    actor_entropies[ag].append(float(entropy.mean().item()))
                    actor_kls[ag].append(float(approx_kl))
                    actor_clip_fracs[ag].append(float(clip_frac))
                    actor_grad_norms[ag].append(float(gn.item() if hasattr(gn, "item") else gn))

        # Re-evaluate updated policy pi_new^(i_k) over full batch and update cumulative factor F^(k+1)
        with torch.no_grad():
            post_log_prob, _ = actor.evaluate_actions(ag_obs, ag_act)
            factor, step_ratio = compute_cumulative_importance_factor(
                current_factor=factor_k,
                old_log_probs=ag_old_lp,
                new_log_probs=post_log_prob,
            )
            factor_out_means[ag] = float(factor.mean().item())
            post_ratio_means[ag] = float(step_ratio.mean().item())
            f_np = factor.detach().cpu().numpy()
            finite_mask = np.isfinite(f_np)
            nonfinite_cnt = int(np.size(f_np) - np.count_nonzero(finite_mask))
            f_valid = f_np[finite_mask] if np.any(finite_mask) else np.array([np.nan], dtype=np.float32)
            extreme_frac = float(np.mean((f_valid < 0.1) | (f_valid > 10.0))) if np.any(finite_mask) else 1.0
            factor_stats_per_agent[ag] = {
                "factor_mean": float(np.mean(f_valid)),
                "factor_std": float(np.std(f_valid)),
                "factor_median": float(np.median(f_valid)),
                "factor_p95": float(np.percentile(f_valid, 95)),
                "factor_p99": float(np.percentile(f_valid, 99)),
                "factor_max": float(np.max(f_valid)),
                "factor_min": float(np.min(f_valid)),
                "nonfinite_count": nonfinite_cnt,
                "extreme_frac_outside_01_10": extreme_frac,
            }

    # =====================================================================
    # 2. Centralized Critic Update on Global States
    # =====================================================================
    critic_losses: List[float] = []
    for _ in range(ppo_epochs):
        for mb_np in HAPPORolloutBuffer.iter_minibatch_indices(num_steps, minibatch_size, rng=rng):
            mb_idx = torch.as_tensor(mb_np, dtype=torch.long, device=agent.device)
            _, value_loss = agent.critic.evaluate_values(
                state=global_states[mb_idx],
                old_values=old_values[mb_idx],
                returns=returns[mb_idx],
            )
            total_critic_loss = value_loss_coef * value_loss

            agent.critic_optimizer.zero_grad()
            total_critic_loss.backward()
            nn.utils.clip_grad_norm_(agent.critic.parameters(), max_grad_norm)
            agent.critic_optimizer.step()
            critic_losses.append(float(value_loss.item()))

    metrics: Dict[str, Any] = {
        "update_order": list(order),
        "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
        "actor_loss_mean": float(np.mean([v for vals in actor_losses.values() for v in vals])) if any(actor_losses.values()) else 0.0,
        "entropy_mean": float(np.mean([v for vals in actor_entropies.values() for v in vals])) if any(actor_entropies.values()) else 0.0,
        "grad_norm_mean": float(np.mean([v for vals in actor_grad_norms.values() for v in vals])) if any(actor_grad_norms.values()) else 0.0,
        "final_factor_mean": float(factor.mean().item()),
        "final_factor_std": float(factor.std().item()),
        "final_factor_median": float(factor.median().item()),
        "final_factor_p95": float(np.percentile(factor.detach().cpu().numpy(), 95)),
        "final_factor_p99": float(np.percentile(factor.detach().cpu().numpy(), 99)),
        "final_factor_max": float(factor.max().item()),
        "final_factor_min": float(factor.min().item()),
        "final_nonfinite_count": int(sum(v["nonfinite_count"] for v in factor_stats_per_agent.values())),
        "factor_stats_per_agent": factor_stats_per_agent,
    }
    for ag in order:
        metrics[f"actor_loss_{ag}"] = float(np.mean(actor_losses[ag])) if actor_losses[ag] else 0.0
        metrics[f"entropy_{ag}"] = float(np.mean(actor_entropies[ag])) if actor_entropies[ag] else 0.0
        metrics[f"approx_kl_{ag}"] = float(np.mean(actor_kls[ag])) if actor_kls[ag] else 0.0
        metrics[f"clip_fraction_{ag}"] = float(np.mean(actor_clip_fracs[ag])) if actor_clip_fracs[ag] else 0.0
        metrics[f"grad_norm_{ag}"] = float(np.mean(actor_grad_norms[ag])) if actor_grad_norms[ag] else 0.0
        metrics[f"factor_in_mean_{ag}"] = factor_in_means[ag]
        metrics[f"factor_out_mean_{ag}"] = factor_out_means[ag]
        metrics[f"post_ratio_mean_{ag}"] = post_ratio_means[ag]
    return metrics
