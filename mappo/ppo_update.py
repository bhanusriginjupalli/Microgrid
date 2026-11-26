"""
mappo/ppo_update.py
============================================================
MAPPO policy and value optimization routines:
Clipped surrogate objective, entropy bonus, value loss,
gradient clipping, KL divergence and clip fraction monitoring.
============================================================
"""

from typing import Dict, Any, List
import torch
import torch.nn as nn
from torch.optim import Adam

from mappo.critic import CentralizedValueFunction
from mappo.rollout_buffer import MultiAgentRolloutBuffer


def update_mappo(
    actors: Dict[str, nn.Module],
    critic: CentralizedValueFunction,
    actor_optimizers: Dict[str, Adam],
    critic_optimizer: Adam,
    buffer: MultiAgentRolloutBuffer,
    ppo_epochs: int = 10,
    minibatch_size: int = 256,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    value_loss_coef: float = 0.5,
    max_grad_norm: float = 0.5,
) -> Dict[str, float]:
    """
    Execute PPO optimization over the collected rollout transitions.
    """
    actor_losses = {ag: [] for ag in actors.keys()}
    actor_entropies = {ag: [] for ag in actors.keys()}
    actor_kls = {ag: [] for ag in actors.keys()}
    actor_clip_fracs = {ag: [] for ag in actors.keys()}
    critic_losses = []

    for epoch in range(ppo_epochs):
        for batch in buffer.get_batches(minibatch_size):
            obs = batch["obs"]
            proposed_actions = batch["proposed_actions"]
            old_log_probs = batch["old_log_probs"]
            global_states = batch["global_states"]
            old_values = batch["old_values"]
            returns = batch["returns"]
            advantages = batch["advantages"]

            # ========================================================
            # 1. Update Decentralized Actors
            # ========================================================
            for ag, actor in actors.items():
                ag_obs = obs[ag]
                ag_act = proposed_actions[ag]
                ag_old_lp = old_log_probs[ag]

                # Evaluate proposed actions under current policy
                new_log_prob, entropy = actor.evaluate_actions(ag_obs, ag_act)

                # Likelihood ratio: r(theta) = exp(new_log_prob - old_log_prob)
                ratio = torch.exp(new_log_prob - ag_old_lp)

                # Clipped surrogate objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus
                entropy_loss = -entropy_coef * entropy.mean()

                total_actor_loss = policy_loss + entropy_loss

                # Backward and step actor optimizer
                actor_optimizers[ag].zero_grad()
                total_actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                actor_optimizers[ag].step()

                # Telemetry
                with torch.no_grad():
                    approx_kl = (ag_old_lp - new_log_prob).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()
                    actor_losses[ag].append(policy_loss.item())
                    actor_entropies[ag].append(entropy.mean().item())
                    actor_kls[ag].append(approx_kl)
                    actor_clip_fracs[ag].append(clip_frac)

            # ========================================================
            # 2. Update Centralized Critic
            # ========================================================
            _, value_loss = critic.evaluate_values(
                state=global_states,
                old_values=old_values,
                returns=returns,
            )

            total_critic_loss = value_loss_coef * value_loss

            critic_optimizer.zero_grad()
            total_critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optimizer.step()

            critic_losses.append(value_loss.item())

    # Aggregate metrics
    metrics = {
        "critic_loss": float(torch.tensor(critic_losses).mean().item()) if critic_losses else 0.0,
    }
    for ag in actors.keys():
        metrics[f"actor_loss_{ag}"] = float(torch.tensor(actor_losses[ag]).mean().item()) if actor_losses[ag] else 0.0
        metrics[f"entropy_{ag}"] = float(torch.tensor(actor_entropies[ag]).mean().item()) if actor_entropies[ag] else 0.0
        metrics[f"approx_kl_{ag}"] = float(torch.tensor(actor_kls[ag]).mean().item()) if actor_kls[ag] else 0.0
        metrics[f"clip_fraction_{ag}"] = float(torch.tensor(actor_clip_fracs[ag]).mean().item()) if actor_clip_fracs[ag] else 0.0

    return metrics
