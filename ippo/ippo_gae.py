"""
ippo/ippo_gae.py
================
Per-agent Generalized Advantage Estimation (GAE) for IPPO.
Each agent i computes its own advantages A_i and returns R_i using only the
shared step reward and its independent critic V_i(o_i), with strict episode
boundary isolation and independent per-agent advantage normalization.
"""

from typing import Dict, Tuple
import numpy as np

from ippo.ippo_buffer import IPPORolloutBuffer


def compute_gae_sequence(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    truncated: np.ndarray,
    next_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute un-normalized GAE advantages and returns for one single-episode sequence."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    if T == 0:
        return advantages, np.zeros(0, dtype=np.float32)

    last_gae_lam = 0.0
    for t in reversed(range(T)):
        is_terminal = bool(dones[t]) and not bool(truncated[t])
        non_terminal = 0.0 if is_terminal else 1.0
        next_v = float(next_value) if t == T - 1 else float(values[t + 1])

        delta = float(rewards[t]) + gamma * next_v * non_terminal - float(values[t])
        last_gae_lam = delta + gamma * gae_lambda * non_terminal * last_gae_lam
        advantages[t] = last_gae_lam

    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


def finalize_episode_ippo_gae(
    buffer: IPPORolloutBuffer,
    next_values: Dict[str, float],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> None:
    """
    Compute per-agent GAE over the current episode slice [episode_start_idx : step_idx]
    without bootstrapping across episode boundaries.
    """
    start = buffer.episode_start_idx
    end = buffer.step_idx
    if end <= start:
        return

    rewards_ep = buffer.rewards[start:end]
    dones_ep = buffer.dones[start:end]
    trunc_ep = buffer.truncated[start:end]

    for ag in buffer.agents:
        vals_ep = buffer.values[ag][start:end]
        adv_ep, ret_ep = compute_gae_sequence(
            rewards=rewards_ep,
            values=vals_ep,
            dones=dones_ep,
            truncated=trunc_ep,
            next_value=float(next_values.get(ag, 0.0)),
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        buffer.advantages[ag][start:end] = adv_ep
        buffer.returns[ag][start:end] = ret_ep

    buffer.episode_start_idx = end


def normalize_ippo_advantages(buffer: IPPORolloutBuffer) -> None:
    """Independently normalize stored advantages per agent to zero mean and unit std."""
    T = buffer.step_idx
    if T == 0:
        return
    for ag in buffer.agents:
        adv_slice = buffer.advantages[ag][:T]
        mean = float(np.mean(adv_slice))
        std = float(np.std(adv_slice)) + 1e-8
        buffer.advantages[ag][:T] = (adv_slice - mean) / std


def compute_ippo_gae(
    buffer: IPPORolloutBuffer,
    next_values: Dict[str, float],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize_advantages: bool = True,
) -> None:
    """Convenience wrapper for single-trajectory buffers matching mappo.gae.compute_gae."""
    finalize_episode_ippo_gae(
        buffer=buffer,
        next_values=next_values,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    if normalize_advantages:
        normalize_ippo_advantages(buffer)
