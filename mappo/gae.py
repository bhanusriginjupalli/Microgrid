"""
mappo/gae.py
============================================================
Generalized Advantage Estimation (GAE) for Multi-Agent PPO.
Handles episode terminations, truncations, and advantage normalization.
============================================================
"""

from typing import Tuple
import numpy as np

from mappo.rollout_buffer import MultiAgentRolloutBuffer


def compute_gae(
    buffer: MultiAgentRolloutBuffer,
    next_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize_advantages: bool = True,
):
    """
    Compute GAE advantages and discounted returns across the buffer trajectory.

    Args:
        buffer: MultiAgentRolloutBuffer instance
        next_value: Critic estimate V(s_next) of the state following the last step
        gamma: Discount factor
        gae_lambda: GAE exponential decay parameter
        normalize_advantages: Whether to normalize advantages to mean 0, std 1
    """
    T = buffer.step_idx
    if T == 0:
        return

    advantages = np.zeros(T, dtype=np.float32)
    last_gae_lam = 0.0

    # Backward recursion through time
    for t in reversed(range(T)):
        if t == T - 1:
            # If the episode truncated at horizon, bootstrap with next_value
            # If terminated naturally, done mask applies
            next_v = next_value
            non_terminal = 1.0 - float(buffer.dones[t] and not buffer.truncated[t])
        else:
            next_v = buffer.values[t + 1]
            non_terminal = 1.0 - float(buffer.dones[t] and not buffer.truncated[t])

        # TD residual: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = buffer.rewards[t] + gamma * next_v * non_terminal - buffer.values[t]

        # GAE accumulation: A_t = delta_t + gamma * lambda * (1 - done) * A_{t+1}
        last_gae_lam = delta + gamma * gae_lambda * non_terminal * last_gae_lam
        advantages[t] = last_gae_lam

    # Returns: R_t = A_t + V(s_t)
    returns = advantages + buffer.values[:T]

    # Save to buffer
    buffer.advantages[:T] = advantages
    buffer.returns[:T] = returns

    # Optional advantage normalization
    if normalize_advantages:
        adv_slice = buffer.advantages[:T]
        mean = np.mean(adv_slice)
        std = np.std(adv_slice) + 1e-8
        buffer.advantages[:T] = (adv_slice - mean) / std
