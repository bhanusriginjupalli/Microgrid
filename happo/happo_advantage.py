"""
happo/happo_advantage.py
========================
Centralized GAE estimation and Canonical HAPPO Advantage Decomposition helpers
following Kuba et al. (ICLR 2022):
  - Joint GAE A_hat(s, a) from centralized V_phi(s)
  - Cumulative importance ratio factor F^(k) = prod_{j < k} pi_new^j / pi_old^j
  - Sequential effective advantage M^(k) = F^(k) * A_hat(s, a)
"""

from typing import Tuple
import numpy as np
import torch

from happo.happo_buffer import HAPPORolloutBuffer


def compute_happo_gae_sequence(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    truncated: np.ndarray,
    next_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute un-normalized centralized GAE advantages and returns for one episode."""
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


def finalize_episode_happo_gae(
    buffer: HAPPORolloutBuffer,
    next_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> None:
    """Compute centralized GAE on the active episode slice without cross-episode leak."""
    start = buffer.episode_start_idx
    end = buffer.step_idx
    if end <= start:
        return

    adv_ep, ret_ep = compute_happo_gae_sequence(
        rewards=buffer.rewards[start:end],
        values=buffer.values[start:end],
        dones=buffer.dones[start:end],
        truncated=buffer.truncated[start:end],
        next_value=float(next_value),
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    buffer.advantages[start:end] = adv_ep
    buffer.returns[start:end] = ret_ep
    buffer.episode_start_idx = end


def normalize_happo_advantages(buffer: HAPPORolloutBuffer) -> None:
    """Normalize centralized advantages across the active rollout batch."""
    T = buffer.step_idx
    if T == 0:
        return
    adv_slice = buffer.advantages[:T]
    mean = float(np.mean(adv_slice))
    std = float(np.std(adv_slice)) + 1e-8
    buffer.advantages[:T] = (adv_slice - mean) / std


def compute_decomposed_advantage(
    joint_advantages: torch.Tensor,
    cumulative_factor: torch.Tensor,
) -> torch.Tensor:
    """
    Canonical HAPPO effective sequential advantage M^(i_k) = F^(k) * A_hat(s, a),
    detached from previously updated agent policies to enforce strict gradient isolation.
    """
    return cumulative_factor.detach() * joint_advantages


def compute_cumulative_importance_factor(
    current_factor: torch.Tensor,
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Update cumulative importance ratio after agent i_k completes its update:
        r_post^(i_k) = exp(log pi_new^(i_k) - log pi_old^(i_k))
        F^(k+1) = F^(k) * r_post^(i_k)

    Returns:
        (updated_factor, step_agent_ratio)
    """
    with torch.no_grad():
        step_agent_ratio = torch.exp(new_log_probs - old_log_probs)
        updated_factor = current_factor.detach() * step_agent_ratio
    return updated_factor.detach(), step_agent_ratio.detach()
