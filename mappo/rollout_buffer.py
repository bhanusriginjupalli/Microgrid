"""
mappo/rollout_buffer.py
============================================================
Multi-Agent Rollout Buffer for MAPPO.
Stores proposed actions, executed actions, per-agent log probabilities,
centralized global states, value predictions, and shield diagnostics.
============================================================
"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch


class MultiAgentRolloutBuffer:
    """
    Rollout storage supporting CTDE Multi-Agent PPO.
    Correctly maintains proposed action likelihoods while tracking
    executed shielded actions for downstream diagnostics.
    """

    def __init__(
        self,
        agents: List[str],
        buffer_size: int,
        obs_shapes: Dict[str, tuple],
        action_shapes: Dict[str, tuple],
        global_state_dim: int = 26,
        device: str = "cpu",
    ):
        self.agents = agents
        self.buffer_size = buffer_size
        self.obs_shapes = obs_shapes
        self.action_shapes = action_shapes
        self.global_state_dim = global_state_dim
        self.device = torch.device(device)

        self.reset()

    def reset(self):
        """Clear the buffer."""
        self.step_idx = 0
        self.is_full = False

        # Per-agent trajectory arrays
        self.obs: Dict[str, np.ndarray] = {
            ag: np.zeros((self.buffer_size,) + self.obs_shapes[ag], dtype=np.float32)
            for ag in self.agents
        }
        self.proposed_actions: Dict[str, np.ndarray] = {
            ag: np.zeros((self.buffer_size,) + self.action_shapes[ag], dtype=np.float32)
            for ag in self.agents
        }
        self.executed_actions: Dict[str, np.ndarray] = {
            ag: np.zeros((self.buffer_size,) + self.action_shapes[ag], dtype=np.float32)
            for ag in self.agents
        }
        self.log_probs: Dict[str, np.ndarray] = {
            ag: np.zeros((self.buffer_size,), dtype=np.float32)
            for ag in self.agents
        }

        # Centralized and environment transitions
        self.global_states = np.zeros((self.buffer_size, self.global_state_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.values = np.zeros((self.buffer_size,), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size,), dtype=np.bool_)
        self.truncated = np.zeros((self.buffer_size,), dtype=np.bool_)

        # GAE targets
        self.advantages = np.zeros((self.buffer_size,), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size,), dtype=np.float32)

        # Diagnostics
        self.shield_interventions: List[Dict[str, Any]] = []

    def insert(
        self,
        obs: Dict[str, np.ndarray],
        global_state: np.ndarray,
        proposed_actions: Dict[str, np.ndarray],
        executed_actions: Dict[str, np.ndarray],
        log_probs: Dict[str, float],
        reward: float,
        value: float,
        done: bool,
        truncated: bool,
        shield_info: Optional[Dict[str, Any]] = None,
    ):
        """Insert a single step transition into buffer."""
        idx = self.step_idx
        assert idx < self.buffer_size, "RolloutBuffer overflow: step exceeds buffer capacity."

        for ag in self.agents:
            self.obs[ag][idx] = obs[ag]
            self.proposed_actions[ag][idx] = np.asarray(proposed_actions[ag], dtype=np.float32).reshape(self.action_shapes[ag])
            self.executed_actions[ag][idx] = np.asarray(executed_actions[ag], dtype=np.float32).reshape(self.action_shapes[ag])
            self.log_probs[ag][idx] = float(log_probs[ag])

        self.global_states[idx] = np.asarray(global_state, dtype=np.float32)
        self.rewards[idx] = float(reward)
        self.values[idx] = float(value)
        self.dones[idx] = bool(done)
        self.truncated[idx] = bool(truncated)

        if shield_info is not None:
            self.shield_interventions.append(shield_info)

        self.step_idx += 1
        if self.step_idx == self.buffer_size:
            self.is_full = True

    def get_batches(
        self,
        batch_size: int,
    ):
        """
        Yield randomized minibatches of transitions for PPO training.
        """
        total_steps = self.step_idx
        indices = np.random.permutation(total_steps)

        for start in range(0, total_steps, batch_size):
            end = min(start + batch_size, total_steps)
            mb_inds = indices[start:end]

            batch_obs = {
                ag: torch.as_tensor(self.obs[ag][mb_inds], device=self.device)
                for ag in self.agents
            }
            batch_proposed_actions = {
                ag: torch.as_tensor(self.proposed_actions[ag][mb_inds], device=self.device)
                for ag in self.agents
            }
            batch_old_log_probs = {
                ag: torch.as_tensor(self.log_probs[ag][mb_inds], device=self.device)
                for ag in self.agents
            }

            batch_global_states = torch.as_tensor(self.global_states[mb_inds], device=self.device)
            batch_old_values = torch.as_tensor(self.values[mb_inds], device=self.device)
            batch_returns = torch.as_tensor(self.returns[mb_inds], device=self.device)
            batch_advantages = torch.as_tensor(self.advantages[mb_inds], device=self.device)

            yield {
                "obs": batch_obs,
                "proposed_actions": batch_proposed_actions,
                "old_log_probs": batch_old_log_probs,
                "global_states": batch_global_states,
                "old_values": batch_old_values,
                "returns": batch_returns,
                "advantages": batch_advantages,
            }
