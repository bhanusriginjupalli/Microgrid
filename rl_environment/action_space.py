# GRIDGUARD AI
# Phase 2: Action Space Definitions & Mapping
# ============================================================
# Action spaces for the 5 decentralized microgrid agents:
#   1. renewable: Box(2,) -> [solar_curtailment, wind_curtailment] in [0, 1]
#   2. battery: Box(1,) -> power command in [-1, 1] (+ charge, - discharge)
#   3. generator: Box(1,) -> requested capacity fraction in [0, 1]
#   4. load_management: Discrete(6) -> curtailment level [0..5]
#   5. grid_resilience: Discrete(5) -> grid operational command [0..4]
# ============================================================

from typing import Dict, Any, Tuple
import numpy as np
from gymnasium import spaces

from rl_environment.agent_config import AGENTS, CAPACITY_SCALING

GRID_ACTIONS = {
    0: "STAY_CONNECTED",
    1: "REQUEST_ISLANDING",
    2: "STAY_ISLANDED",
    3: "REQUEST_RECONNECTION",
    4: "RECOVERY_MODE",
}

LOAD_ACTIONS = {
    0: "NO_CURTAILMENT",
    1: "FLEXIBLE_25_PCT",
    2: "FLEXIBLE_50_PCT",
    3: "FLEXIBLE_75_PCT",
    4: "FLEXIBLE_100_PCT",
    5: "FLEX_100_PLUS_OTHER_SHED",
}

LOAD_CURTAILMENT_FRACTIONS = {
    0: 0.00,  # No curtailment
    1: 0.25,  # 25% flexible load
    2: 0.50,  # 50% flexible load
    3: 0.75,  # 75% flexible load
    4: 1.00,  # 100% flexible load
    5: 1.00,  # 100% flexible load + other load shed
}


def get_agent_action_spaces() -> Dict[str, spaces.Space]:
    """Return dictionary of Gymnasium action spaces for each agent."""
    return {
        "renewable": spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        ),
        "battery": spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        ),
        "generator": spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        ),
        "load_management": spaces.Discrete(6),
        "grid_resilience": spaces.Discrete(5),
    }


def parse_agent_action(agent: str, raw_action: Any) -> Dict[str, Any]:
    """
    Parse and validate raw action into structured physical setpoints.
    """
    if agent == "renewable":
        arr = np.asarray(raw_action, dtype=np.float32).flatten()
        if len(arr) != 2:
            raise ValueError(f"Renewable action must have shape (2,), got {arr.shape}")
        solar_curt = float(np.clip(arr[0], 0.0, 1.0))
        wind_curt = float(np.clip(arr[1], 0.0, 1.0))
        return {
            "solar_curtailment_factor": solar_curt,
            "wind_curtailment_factor": wind_curt,
        }

    elif agent == "battery":
        arr = np.asarray(raw_action, dtype=np.float32).flatten()
        if len(arr) != 1:
            raise ValueError(f"Battery action must have shape (1,), got {arr.shape}")
        cmd = float(np.clip(arr[0], -1.0, 1.0))
        # Sign convention: +1 = max charge (+0.25 MW), -1 = max discharge (-0.25 MW)
        power_req_mw = cmd * CAPACITY_SCALING["battery_power_mw"]
        mode = "CHARGE" if cmd > 0 else ("DISCHARGE" if cmd < 0 else "IDLE")
        return {
            "battery_cmd": cmd,
            "action_normalized": cmd,
            "power_mw": power_req_mw,
            "requested_power_mw": power_req_mw,
            "mode": mode,
        }

    elif agent == "generator":
        arr = np.asarray(raw_action, dtype=np.float32).flatten()
        if len(arr) != 1:
            raise ValueError(f"Generator action must have shape (1,), got {arr.shape}")
        frac = float(np.clip(arr[0], 0.0, 1.0))
        power_req_mw = frac * CAPACITY_SCALING["generator_capacity_mw"]
        return {
            "generator_fraction": frac,
            "power_mw": power_req_mw,
            "requested_power_mw": power_req_mw,
            "should_run": frac > 0.05,
        }

    elif agent == "load_management":
        val = int(raw_action)
        if not 0 <= val <= 5:
            raise ValueError(f"Load management action must be in [0..5], got {val}")
        shed_other = (val == 5)
        return {
            "action_idx": val,
            "command": LOAD_ACTIONS[val],
            "flexible_curtail_frac": LOAD_CURTAILMENT_FRACTIONS[val],
            "shed_other_loads": shed_other,
        }

    elif agent == "grid_resilience":
        val = int(raw_action)
        if not 0 <= val <= 4:
            raise ValueError(f"Grid resilience action must be in [0..4], got {val}")
        return {
            "action_idx": val,
            "command": GRID_ACTIONS[val],
        }

    else:
        raise ValueError(f"Unknown agent: {agent}")


if __name__ == "__main__":
    action_spaces = get_agent_action_spaces()
    print("GRIDGUARD AI - Action Spaces Definition Self-Test")
    print("=" * 50)
    for agent_id, space in action_spaces.items():
        sample = space.sample()
        parsed = parse_agent_action(agent_id, sample)
        print(f"Agent: {agent_id:16s} | Space: {str(space):25s} | Sample: {str(sample):18s} -> Parsed: {parsed}")
    print("Action space self-test passed successfully.")
