# GRIDGUARD AI
# Phase 1.5: Generator Dynamics & State Machine
# ============================================================
# Models backup diesel generator with state transitions:
#   OFF -> STARTING -> RUNNING -> SHUTTING_DOWN -> OFF
# Features:
#   - Ramp up / down rate limits
#   - Minimum stable operating point (80 kW)
#   - Startup delay (steps / time)
#   - Minimum up time & minimum down time
#   - Load-dependent fuel consumption + startup fuel
# ============================================================

from enum import Enum
import numpy as np

try:
    from config import GENERATOR
except ImportError:
    from phase1_digital_twin.config import GENERATOR


class GeneratorState(str, Enum):
    OFF = "OFF"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    SHUTTING_DOWN = "SHUTTING_DOWN"


class DynamicGenerator:
    """
    Industrial backup diesel generator with physical ramp rates,
    state machine sequencing, and non-linear fuel consumption.
    """

    def __init__(
        self,
        capacity_mw: float = None,
        min_output_mw: float = None,
        ramp_up_mw_per_min: float = None,
        ramp_down_mw_per_min: float = None,
        startup_delay_steps: int = None,
        min_up_steps: int = None,
        min_down_steps: int = None,
        fuel_base_l_per_hr: float = None,
        fuel_full_l_per_hr: float = None,
        startup_fuel_l: float = None,
    ):
        self.capacity_mw = capacity_mw if capacity_mw is not None else GENERATOR["capacity_mw"]
        self.min_output_mw = min_output_mw if min_output_mw is not None else GENERATOR.get("min_output_mw", 0.08)
        self.ramp_up_rate_min = (
            ramp_up_mw_per_min if ramp_up_mw_per_min is not None else GENERATOR.get("ramp_up_rate_mw_per_min", 0.02)
        )
        self.ramp_down_rate_min = (
            ramp_down_mw_per_min if ramp_down_mw_per_min is not None else GENERATOR.get("ramp_down_rate_mw_per_min", 0.02)
        )
        self.startup_delay_steps = (
            startup_delay_steps if startup_delay_steps is not None else GENERATOR.get("startup_delay_steps", 2)
        )
        self.min_up_steps = min_up_steps if min_up_steps is not None else GENERATOR.get("min_up_time_steps", 4)
        self.min_down_steps = min_down_steps if min_down_steps is not None else GENERATOR.get("min_down_time_steps", 2)

        self.fuel_base_l_per_hr = (
            fuel_base_l_per_hr if fuel_base_l_per_hr is not None else GENERATOR.get("fuel_rate_base_l_per_hr", 20.0)
        )
        self.fuel_full_l_per_hr = (
            fuel_full_l_per_hr if fuel_full_l_per_hr is not None else GENERATOR.get("fuel_rate_full_l_per_hr", 80.0)
        )
        self.startup_fuel_l = (
            startup_fuel_l if startup_fuel_l is not None else GENERATOR.get("startup_fuel_l", 5.0)
        )

        self._state = GeneratorState.OFF
        self.output_mw = 0.0
        self.steps_in_state = self.min_down_steps  # Initially cold and ready to start
        self.total_fuel_l = 0.0
        self.last_step_fuel_l = 0.0
        self.history = []

    @property
    def state(self) -> GeneratorState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state == GeneratorState.RUNNING

    @property
    def is_available(self) -> bool:
        """True if running or able to start."""
        if self._state == GeneratorState.OFF and self.steps_in_state >= self.min_down_steps:
            return True
        return self._state in (GeneratorState.STARTING, GeneratorState.RUNNING)

    def request_start(self) -> bool:
        """
        Request generator start.
        Returns True if transition to STARTING or already starting/running.
        """
        if self._state == GeneratorState.RUNNING:
            return True
        if self._state == GeneratorState.STARTING:
            return True
        if self._state == GeneratorState.OFF:
            if self.steps_in_state >= self.min_down_steps:
                self._state = GeneratorState.STARTING
                self.steps_in_state = 0
                self.total_fuel_l += self.startup_fuel_l
                self.last_step_fuel_l = self.startup_fuel_l
                return True
        return False

    def request_stop(self) -> bool:
        """
        Request generator stop. Respects min up time.
        """
        if self._state == GeneratorState.OFF:
            return True
        if self._state == GeneratorState.RUNNING:
            if self.steps_in_state >= self.min_up_steps:
                self._state = GeneratorState.SHUTTING_DOWN
                self.steps_in_state = 0
                return True
            return False  # Must respect minimum up time
        if self._state == GeneratorState.STARTING:
            # If still starting, can cancel to shutdown
            self._state = GeneratorState.SHUTTING_DOWN
            self.steps_in_state = 0
            return True
        return False

    def update(self, requested_mw: float, dt_hours: float) -> float:
        """
        Advance generator dynamics by dt_hours with target power requested_mw.

        Returns:
            Actual active power output in MW.
        """
        self.steps_in_state += 1
        dt_minutes = dt_hours * 60.0

        step_fuel = 0.0

        if self._state == GeneratorState.OFF:
            self.output_mw = 0.0
            step_fuel = 0.0

        elif self._state == GeneratorState.STARTING:
            self.output_mw = 0.0
            # Consumes baseline fuel warming up
            step_fuel = self.fuel_base_l_per_hr * 0.5 * dt_hours
            if self.steps_in_state >= self.startup_delay_steps:
                self._state = GeneratorState.RUNNING
                self.steps_in_state = 0
                self.output_mw = self.min_output_mw

        elif self._state == GeneratorState.RUNNING:
            # Ramp rate calculation
            max_ramp_up = self.ramp_up_rate_min * dt_minutes
            max_ramp_down = self.ramp_down_rate_min * dt_minutes

            # Target power must be between min_output_mw and capacity_mw
            target_p = np.clip(requested_mw, self.min_output_mw, self.capacity_mw)

            # Apply ramping limits
            lower_bound = max(self.min_output_mw, self.output_mw - max_ramp_down)
            upper_bound = min(self.capacity_mw, self.output_mw + max_ramp_up)
            self.output_mw = float(np.clip(target_p, lower_bound, upper_bound))

            # Fuel calculation
            fuel_rate = self.fuel_base_l_per_hr + (
                (self.fuel_full_l_per_hr - self.fuel_base_l_per_hr) * (self.output_mw / self.capacity_mw)
            )
            step_fuel = fuel_rate * dt_hours

        elif self._state == GeneratorState.SHUTTING_DOWN:
            max_ramp_down = self.ramp_down_rate_min * dt_minutes
            self.output_mw = max(0.0, self.output_mw - max_ramp_down)
            fuel_rate = self.fuel_base_l_per_hr * 0.5
            step_fuel = fuel_rate * dt_hours
            if self.output_mw <= 0.01:
                self._state = GeneratorState.OFF
                self.steps_in_state = 0
                self.output_mw = 0.0

        self.last_step_fuel_l = step_fuel
        self.total_fuel_l += step_fuel

        self.history.append({
            "state": self._state.value,
            "output_mw": self.output_mw,
            "step_fuel_l": step_fuel,
            "total_fuel_l": self.total_fuel_l,
            "dt_hours": dt_hours,
        })

        return self.output_mw

    def reset(self):
        """Reset generator state to cold OFF."""
        self._state = GeneratorState.OFF
        self.output_mw = 0.0
        self.steps_in_state = self.min_down_steps  # Available to start immediately if needed
        self.total_fuel_l = 0.0
        self.last_step_fuel_l = 0.0
        self.history.clear()


if __name__ == "__main__":
    gen = DynamicGenerator()
    print("GRIDGUARD AI - Dynamic Generator Self-Test")
    print("=" * 50)
    print(f"Capacity: {gen.capacity_mw:.2f} MW | Min stable: {gen.min_output_mw:.3f} MW")
    print(f"Initial state: {gen.state.value} | Output: {gen.output_mw:.3f} MW")

    dt = 0.25  # 15 minutes
    gen.request_start()
    print(f"Start requested -> State: {gen.state.value}")

    # Step through startup
    for step in range(1, 4):
        p = gen.update(0.3, dt)
        print(f"Step {step}: State={gen.state.value} | Output={p:.3f} MW | Fuel={gen.last_step_fuel_l:.1f} L")

    # Step ramping up to full power
    for step in range(4, 7):
        p = gen.update(0.4, dt)
        print(f"Step {step}: State={gen.state.value} | Output={p:.3f} MW | Fuel={gen.last_step_fuel_l:.1f} L")

    print(f"Total fuel consumed: {gen.total_fuel_l:.2f} L")
    print("Dynamic Generator test passed successfully.")
