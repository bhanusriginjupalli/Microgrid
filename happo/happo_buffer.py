"""
happo/happo_buffer.py
=====================
Rollout storage for Heterogeneous-Agent PPO (HAPPO).
Stores per-agent local observations, proposed pre-shield actions with exact
continuous/discrete dtypes, old log probabilities under the sampling policy,
and centralized global states, shared rewards, and centralized critic values.
"""

from typing import Dict, Any, Generator
import numpy as np
import torch


class HAPPORolloutBuffer:
    """Fixed-capacity multi-episode rollout buffer supporting sequential HAPPO updates."""

    def __init__(
        self,
        buffer_size: int,
        obs_dims: Dict[str, int],
        action_spaces: Dict[str, Any],
        global_state_dim: int = 45,
        device: str = "cpu",
    ):
        self.buffer_size = int(buffer_size)
        self.obs_dims = dict(obs_dims)
        self.agents = list(obs_dims.keys())
        self.global_state_dim = int(global_state_dim)
        self.device = torch.device(device)

        self.obs: Dict[str, np.ndarray] = {}
        self.proposed_actions: Dict[str, np.ndarray] = {}
        self.log_probs: Dict[str, np.ndarray] = {}

        for ag in self.agents:
            self.obs[ag] = np.zeros((self.buffer_size, self.obs_dims[ag]), dtype=np.float32)
            space = action_spaces[ag]
            if hasattr(space, "shape") and len(space.shape) > 0:
                act_shape = (self.buffer_size,) + tuple(space.shape)
                act_dtype = np.float32
            else:
                act_shape = (self.buffer_size,)
                act_dtype = np.int64
            self.proposed_actions[ag] = np.zeros(act_shape, dtype=act_dtype)
            self.log_probs[ag] = np.zeros(self.buffer_size, dtype=np.float32)

        self.global_states = np.zeros((self.buffer_size, self.global_state_dim), dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.bool_)
        self.truncated = np.zeros(self.buffer_size, dtype=np.bool_)

        self.step_idx = 0
        self.episode_start_idx = 0
        self.is_full = False

    def reset(self) -> None:
        self.step_idx = 0
        self.episode_start_idx = 0
        self.is_full = False

    def add(
        self,
        obs_dict: Dict[str, np.ndarray],
        global_state: np.ndarray,
        proposed_actions_dict: Dict[str, Any],
        log_probs_dict: Dict[str, float],
        reward: float,
        value: float,
        done: bool,
        truncated: bool = False,
    ) -> None:
        if self.step_idx >= self.buffer_size:
            raise RuntimeError("HAPPORolloutBuffer overflow: call reset() between updates.")

        idx = self.step_idx
        for ag in self.agents:
            if ag in obs_dict:
                self.obs[ag][idx] = np.asarray(obs_dict[ag], dtype=np.float32)
            if ag in proposed_actions_dict:
                act_val = proposed_actions_dict[ag]
                if self.proposed_actions[ag].ndim == 1:
                    self.proposed_actions[ag][idx] = int(act_val)
                else:
                    self.proposed_actions[ag][idx] = np.asarray(act_val, dtype=np.float32)
            self.log_probs[ag][idx] = float(log_probs_dict.get(ag, 0.0))

        self.global_states[idx] = np.asarray(global_state, dtype=np.float32)
        self.rewards[idx] = float(reward)
        self.values[idx] = float(value)
        self.dones[idx] = bool(done)
        self.truncated[idx] = bool(truncated)

        self.step_idx += 1
        if self.step_idx == self.buffer_size:
            self.is_full = True

    def get_full_tensors(self) -> Dict[str, Any]:
        """Materialize active rollout transitions [0:step_idx] as tensors on device."""
        T = self.step_idx
        return {
            "num_steps": T,
            "obs": {ag: torch.as_tensor(self.obs[ag][:T], device=self.device) for ag in self.agents},
            "proposed_actions": {
                ag: torch.as_tensor(self.proposed_actions[ag][:T], device=self.device) for ag in self.agents
            },
            "old_log_probs": {ag: torch.as_tensor(self.log_probs[ag][:T], device=self.device) for ag in self.agents},
            "global_states": torch.as_tensor(self.global_states[:T], device=self.device),
            "old_values": torch.as_tensor(self.values[:T], device=self.device),
            "returns": torch.as_tensor(self.returns[:T], device=self.device),
            "advantages": torch.as_tensor(self.advantages[:T], device=self.device),
        }

    @staticmethod
    def iter_minibatch_indices(
        total_steps: int,
        batch_size: int,
        rng: np.random.Generator | None = None,
    ) -> Generator[np.ndarray, None, None]:
        """Yield shuffled minibatch index arrays for one epoch."""
        perm = rng.permutation(total_steps) if rng is not None else np.random.permutation(total_steps)
        for start in range(0, total_steps, batch_size):
            yield perm[start : min(start + batch_size, total_steps)]
