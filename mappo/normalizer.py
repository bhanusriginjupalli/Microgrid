"""
mappo/normalizer.py
============================================================
Running Mean and Standard Deviation Normalizer for observations
and centralized global states.
============================================================
"""

from typing import Optional, Union, Dict
import numpy as np
import torch
import torch.nn as nn


class RunningMeanStd:
    """Dynamically calculates running empirical mean and variance."""

    def __init__(self, shape: tuple = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if len(x.shape) > len(self.mean.shape) else 1

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class StateNormalizer:
    """Wrapper normalizing observation dicts and global state vector."""

    def __init__(self, enabled: bool = False, clip_limit: float = 10.0):
        self.enabled = enabled
        self.clip_limit = clip_limit
        self.obs_rms: Dict[str, RunningMeanStd] = {}
        self.state_rms: Optional[RunningMeanStd] = None

    def normalize_obs(self, obs_dict: Dict[str, np.ndarray], update: bool = True) -> Dict[str, np.ndarray]:
        if not self.enabled:
            return obs_dict

        norm_obs = {}
        for agent, o in obs_dict.items():
            if agent not in self.obs_rms:
                self.obs_rms[agent] = RunningMeanStd(shape=o.shape)
            if update:
                self.obs_rms[agent].update(o)

            rms = self.obs_rms[agent]
            norm = (o - rms.mean) / np.sqrt(rms.var + 1e-8)
            norm_obs[agent] = np.clip(norm, -self.clip_limit, self.clip_limit).astype(np.float32)

        return norm_obs

    def normalize_state(self, state: np.ndarray, update: bool = True) -> np.ndarray:
        if not self.enabled:
            return state

        if self.state_rms is None:
            self.state_rms = RunningMeanStd(shape=state.shape)
        if update:
            self.state_rms.update(state)

        norm = (state - self.state_rms.mean) / np.sqrt(self.state_rms.var + 1e-8)
        return np.clip(norm, -self.clip_limit, self.clip_limit).astype(np.float32)
