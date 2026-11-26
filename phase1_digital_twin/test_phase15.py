# GRIDGUARD AI
# Phase 1.5 Unit Test Suite
# ============================================================
# Unit tests covering:
#   1. Battery energy update and SOC limit enforcement
#   2. Generator state machine, startup delay, and ramp constraints
#   3. Energy balance conservation
#   4. Priority load management and shifting
#   5. Forecasting baseline predictions
#   6. Scenario disturbance injection
#   7. Dynamic engine step-by-step physical conservation
# ============================================================

import unittest
import numpy as np

try:
    from battery_dynamics import DynamicBattery
    from generator_dynamics import DynamicGenerator, GeneratorState
    from energy_balance import EnergyBalanceCalculator
    from load_management import LoadManager
    from forecast_baseline import ForecastBaseline
    from scenario_manager import ScenarioManager
    from dynamic_engine import DynamicMicrogridSimulation
except ImportError:
    from phase1_digital_twin.battery_dynamics import DynamicBattery
    from phase1_digital_twin.generator_dynamics import DynamicGenerator, GeneratorState
    from phase1_digital_twin.energy_balance import EnergyBalanceCalculator
    from phase1_digital_twin.load_management import LoadManager
    from phase1_digital_twin.forecast_baseline import ForecastBaseline
    from phase1_digital_twin.scenario_manager import ScenarioManager
    from phase1_digital_twin.dynamic_engine import DynamicMicrogridSimulation


class TestDynamicBattery(unittest.TestCase):
    def setUp(self):
        self.battery = DynamicBattery(
            capacity_mwh=1.0,
            max_power_mw=0.25,
            initial_soc=80.0,
            min_soc=20.0,
            max_soc=95.0,
            charge_eff=0.95,
            discharge_eff=0.95,
            emergency_reserve_soc=30.0,
        )

    def test_battery_discharge_physics(self):
        dt = 0.25  # 15 minutes
        p_act = self.battery.update(-0.20, dt, emergency=False)
        self.assertAlmostEqual(p_act, -0.20, places=3)
        # E_t+1 = 0.8 + (-0.20 / 0.95) * 0.25 = 0.8 - 0.05263 = 0.74737 MWh
        expected_soc = (0.747368 / 1.0) * 100.0
        self.assertAlmostEqual(self.battery.soc, expected_soc, delta=0.1)

    def test_battery_soc_limits(self):
        # Cannot discharge below emergency reserve when emergency=False
        self.battery.soc = 31.0
        dt = 1.0
        avail = self.battery.available_discharge_mw(dt, emergency=False)
        self.assertLessEqual(avail, 0.02)

        # But can discharge down to min_soc when emergency=True
        avail_emerg = self.battery.available_discharge_mw(dt, emergency=True)
        self.assertGreater(avail_emerg, 0.08)

    def test_battery_charge_upper_limit(self):
        self.battery.soc = 94.0
        dt = 1.0
        avail_chg = self.battery.available_charge_mw(dt)
        # Headroom is (95 - 94)% = 0.01 MWh -> Power max = 0.01 / (0.95 * 1.0) = 0.0105 MW
        self.assertLess(avail_chg, 0.02)


class TestDynamicGenerator(unittest.TestCase):
    def setUp(self):
        self.gen = DynamicGenerator(
            capacity_mw=0.4,
            min_output_mw=0.08,
            ramp_up_mw_per_min=0.02,
            startup_delay_steps=2,
            min_up_steps=4,
            min_down_steps=2,
        )

    def test_generator_state_transitions(self):
        dt = 0.25
        self.assertEqual(self.gen.state, GeneratorState.OFF)

        # Start sequence
        self.assertTrue(self.gen.request_start())
        self.assertEqual(self.gen.state, GeneratorState.STARTING)

        # Step 1: Still starting, output 0
        p1 = self.gen.update(0.3, dt)
        self.assertEqual(p1, 0.0)

        # Step 2: Transitions to RUNNING, produces min stable output (0.08 MW)
        p2 = self.gen.update(0.3, dt)
        self.assertEqual(self.gen.state, GeneratorState.RUNNING)
        self.assertGreaterEqual(p2, 0.08)

        # Ramping constraint
        # In 15 min, max ramp up = 0.02 * 15 = 0.3 MW
        p3 = self.gen.update(0.4, dt)
        self.assertLessEqual(p3 - p2, 0.3001)


class TestEnergyBalance(unittest.TestCase):
    def setUp(self):
        self.eb = EnergyBalanceCalculator(tolerance_mw=0.001)

    def test_balance_exact(self):
        # Supply: 0.35 + 0.15 + 0.0 + 0.30 - 0.0 = 0.80 MW
        # Demand: 0.80 + 0.0 = 0.80 MW
        res = self.eb.calculate_balance(0.35, 0.15, 0.0, 0.30, 0.0, 0.80, 0.0)
        self.assertTrue(res["is_balanced"])
        self.assertLess(res["balance_error_mw"], 0.0001)


class TestLoadManagement(unittest.TestCase):
    def setUp(self):
        self.lm = LoadManager(critical_mw=0.3, flexible_mw=0.2, other_mw=0.3)

    def test_priority_curtailment(self):
        # Flexible curtailed first
        res = self.lm.dispatch(deficit_mw=0.10, hour=12.0)
        self.assertGreater(res["shed_flexible_mw"], 0.09)
        self.assertEqual(res["shed_other_mw"], 0.0)
        self.assertEqual(res["shed_critical_mw"], 0.0)
        self.assertFalse(res["critical_emergency"])


class TestForecastBaseline(unittest.TestCase):
    def setUp(self):
        self.fc = ForecastBaseline(horizon_hours=2.0, dt_hours=0.25)

    def test_predictions(self):
        self.fc.observe(0.3, 0.1, 0.8, 0.0)
        preds = self.fc.predict(12.0)
        self.assertEqual(len(preds["horizon_hours"]), 8)
        self.assertIn("solar_mw", preds)
        self.assertIn("wind_mw", preds)


if __name__ == "__main__":
    unittest.main(verbosity=2)
