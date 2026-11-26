"""
ippo/ippo_buffer.py
===================
Decentralized per-agent rollout storage for IPPO.
Stores local observations, proposed actions, old log probabilities, and
per-agent value estimates V_i(o_i), returns, and advantages. Intentionally
omits global_state storage so IPPO cannot accidentally consume centralized state.
"""

from typing import Dict, Any, Generator
import numpy as np
import torch


class IPPORolloutBuffer:
    """Fixed-capacity multi-episode rollout buffer with per-agent GAE slices."""

    def __init__(
        self,
        buffer_size: int,
        obs_dims: Dict[str, int],
        action_spaces: Dict[str, Any],
        device: str = "cpu",
    ):
        self.buffer_size = int(buffer_size)
        self.obs_dims = dict(obs_dims)
        self.agents = list(obs_dims.keys())
        self.device = torch.device(device)

        self.obs: Dict[str, np.ndarray] = {}
        self.proposed_actions: Dict[str, np.ndarray] = {}
        self.log_probs: Dict[str, np.ndarray] = {}
        self.values: Dict[str, np.ndarray] = {}
        self.returns: Dict[str, np.ndarray] = {}
        self.advantages: Dict[str, np.ndarray] = {}

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
            self.values[ag] = np.zeros(self.buffer_size, dtype=np.float32)
            self.returns[ag] = np.zeros(self.buffer_size, dtype=np.float32)
            self.advantages[ag] = np.zeros(self.buffer_size, dtype=np.float32)

        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
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
        proposed_actions_dict: Dict[str, Any],
        log_probs_dict: Dict[str, float],
        reward: float,
        values_dict: Dict[str, float],
        done: bool,
        truncated: bool = False,
    ) -> None:
        if self.step_idx >= self.buffer_size:
            raise RuntimeError("IPPORolloutBuffer overflow: call reset() between updates.")

        idx = self.step_idx
        for ag in self.agents:
            if ag in obs_dict:
                self.obs[ag][idx] = np.asarray(obs_dict[ag], dtype=np.float32)
            if ag in proposed_actions_dict:
                act_val = proposed_actions_dict[ag]
                if isinstance(self.proposed_actions[ag][idx], np.integer) or self.proposed_actions[ag].ndim == 1:
                    self.proposed_actions[ag][idx] = int(act_val)
                else:
                    self.proposed_actions[ag][idx] = np.asarray(act_val, dtype=np.float32)
            self.log_probs[ag][idx] = float(log_probs_dict.get(ag, 0.0))
            self.values[ag][idx] = float(values_dict.get(ag, 0.0))

        self.rewards[idx] = float(reward)
        self.dones[idx] = bool(done)
        self.truncated[idx] = bool(truncated)

        self.step_idx += 1
        if self.step_idx == self.buffer_size:
            self.is_full = True

    def get_batches(
        self,
        batch_size: int,
    ) -> Generator[Dict[str, Any], None, None]:
        """Yield randomized minibatches containing strictly per-agent decentralized data."""
        total_steps = self.step_idx
        indices = np.random.permutation(total_steps)

        for start in range(0, total_steps, batch_size):
            mb_inds = indices[start : min(start + batch_size, total_steps)]
            if len(mb_inds) == 0:
                continue

            yield {
                "obs": {
                    ag: torch.as_tensor(self.obs[ag][mb_inds], device=self.device)
                    for ag in self.agents
                },
                "proposed_actions": {
                    ag: torch.as_tensor(self.proposed_actions[ag][mb_inds], device=self.device)
                    for ag in self.agents
                },
                "old_log_probs": {
                    ag: torch.as_tensor(self.log_probs[ag][mb_inds], device=self.device)
                    for ag in self.agents
                },
                "old_values": {
                    ag: torch.as_tensor(self.values[ag][mb_inds], device=self.device)
                    for ag in self.agents
                },
                "returns": {
                    ag: torch.as_tensor(self.returns[ag][mb_inds], device=self.device)
                    for ag in self.agents
                },
                "advantages": {
                    ag: torch.as_tensor(self.advantages[ag][mb_inds], device=self.device)
                    for ag in self.agents
                },
            }
