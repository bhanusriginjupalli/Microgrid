# GRIDGUARD AI - Phase 2 RL Environment Package
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PHASE1 = _ROOT / "phase1_digital_twin"
for p in [str(_ROOT), str(_PHASE1)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rl_environment.agent_config import AGENTS, OBS_SHAPES, GLOBAL_STATE_SHAPE
from rl_environment.microgrid_parallel_env import MicrogridParallelEnv
from rl_environment.observation_builder import ObservationBuilder
from rl_environment.action_space import get_agent_action_spaces, parse_agent_action
from rl_environment.reward_function import MicrogridRewardFunction
from rl_environment.safety_shield import SafetyShield
from rl_environment.scenario_sampler import ScenarioSampler

__all__ = [
    "AGENTS",
    "OBS_SHAPES",
    "GLOBAL_STATE_SHAPE",
    "MicrogridParallelEnv",
    "ObservationBuilder",
    "get_agent_action_spaces",
    "parse_agent_action",
    "MicrogridRewardFunction",
    "SafetyShield",
    "ScenarioSampler",
]
