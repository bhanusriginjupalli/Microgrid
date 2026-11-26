"""
ippo/ippo_update.py
===================
Independent PPO (IPPO) actor and critic optimization routine.
Each agent updates its policy using its own decentralized advantage A_i and
updates its own independent critic V_i(o_i) using local observations only.
"""

from typing import Dict
import numpy as np
import torch
import torch.nn as nn

from ippo.ippo_agent import IPPOAgent
from ippo.ippo_buffer import IPPORolloutBuffer


def update_ippo(
    agent: IPPOAgent,
    buffer: IPPORolloutBuffer,
    ppo_epochs: int = 6,
    minibatch_size: int = 128,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.015,
    value_loss_coef: float = 0.5,
    max_grad_norm: float = 0.5,
) -> Dict[str, float]:
    """Execute independent per-agent PPO updates over buffer minibatches."""
    actor_losses = {ag: [] for ag in agent.actors.keys()}
    critic_losses = {ag: [] for ag in agent.actors.keys()}
    actor_entropies = {ag: [] for ag in agent.actors.keys()}
    actor_kls = {ag: [] for ag in agent.actors.keys()}
    actor_clip_fracs = {ag: [] for ag in agent.actors.keys()}

    for _ in range(ppo_epochs):
        for batch in buffer.get_batches(minibatch_size):
            if len(next(iter(batch["advantages"].values()))) < 4 and buffer.step_idx >= 8:
                continue

            for ag, actor in agent.actors.items():
                ag_obs = batch["obs"][ag]
                ag_act = batch["proposed_actions"][ag]
                ag_old_lp = batch["old_log_probs"][ag]
                ag_old_val = batch["old_values"][ag]
                ag_ret = batch["returns"][ag]
                ag_adv = batch["advantages"][ag]

                # 1. Actor update using agent-specific decentralized advantages
                new_log_prob, entropy = actor.evaluate_actions(ag_obs, ag_act)
                ratio = torch.exp(new_log_prob - ag_old_lp)
                surr1 = ratio * ag_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * ag_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy_coef * entropy.mean()
                total_actor_loss = policy_loss + entropy_loss

                actor_opt = agent.actor_optimizers[ag]
                actor_opt.zero_grad()
                total_actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                actor_opt.step()

                # 2. Independent Critic V_i(o_i) update on local observations only
                _, value_loss = agent.critics.evaluate_agent_value(
                    agent=ag,
                    local_obs=ag_obs,
                    old_values=ag_old_val,
                    returns=ag_ret,
                )
                total_critic_loss = value_loss_coef * value_loss

                critic_opt = agent.critics.optimizers[ag]
                critic_opt.zero_grad()
                total_critic_loss.backward()
                nn.utils.clip_grad_norm_(agent.critics.critics[ag].parameters(), max_grad_norm)
                critic_opt.step()

                with torch.no_grad():
                    approx_kl = (ag_old_lp - new_log_prob).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()
                    actor_losses[ag].append(float(policy_loss.item()))
                    critic_losses[ag].append(float(value_loss.item()))
                    actor_entropies[ag].append(float(entropy.mean().item()))
                    actor_kls[ag].append(float(approx_kl))
                    actor_clip_fracs[ag].append(float(clip_frac))

    all_c_losses = [v for vals in critic_losses.values() for v in vals]
    all_a_losses = [v for vals in actor_losses.values() for v in vals]
    all_ents = [v for vals in actor_entropies.values() for v in vals]
    metrics: Dict[str, float] = {
        "critic_loss": float(np.mean(all_c_losses)) if all_c_losses else 0.0,
        "actor_loss_mean": float(np.mean(all_a_losses)) if all_a_losses else 0.0,
        "entropy_mean": float(np.mean(all_ents)) if all_ents else 0.0,
    }
    for ag in agent.actors.keys():
        metrics[f"actor_loss_{ag}"] = float(np.mean(actor_losses[ag])) if actor_losses[ag] else 0.0
        metrics[f"critic_loss_{ag}"] = float(np.mean(critic_losses[ag])) if critic_losses[ag] else 0.0
        metrics[f"entropy_{ag}"] = float(np.mean(actor_entropies[ag])) if actor_entropies[ag] else 0.0
        metrics[f"approx_kl_{ag}"] = float(np.mean(actor_kls[ag])) if actor_kls[ag] else 0.0
        metrics[f"clip_fraction_{ag}"] = float(np.mean(actor_clip_fracs[ag])) if actor_clip_fracs[ag] else 0.0
    return metrics
