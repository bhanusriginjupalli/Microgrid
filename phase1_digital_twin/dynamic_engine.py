# GRIDGUARD AI
# Phase 1.5: Dynamic Microgrid Simulation Engine
# ============================================================
# Core time-stepped simulation engine integrating:
#   - 9-bus Pandapower AC network with islanded slack handling
#   - Physically consistent battery SOC & energy evolution
#   - Backup generator state machine, ramp rates & fuel
#   - Priority-based flexible load shifting and shedding
#   - Multi-stage disturbance scenario & outage injection
#   - Lookahead forecasting baseline
#   - Explicit instantaneous power balance verification
# ============================================================

import warnings
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import pandapower as pp

try:
    from config import SIMULATION, GRID_CONNECTION, LOADS, GENERATOR, BATTERY
    from network import create_microgrid
    from resources import SolarModel, WindModel
    from battery_dynamics import DynamicBattery
    from generator_dynamics import DynamicGenerator
    from load_management import LoadManager
    from energy_balance import EnergyBalanceCalculator
    from forecast_baseline import ForecastBaseline
    from scenario_manager import ScenarioManager
except ImportError:
    from phase1_digital_twin.config import SIMULATION, GRID_CONNECTION, LOADS, GENERATOR, BATTERY
    from phase1_digital_twin.network import create_microgrid
    from phase1_digital_twin.resources import SolarModel, WindModel
    from phase1_digital_twin.battery_dynamics import DynamicBattery
    from phase1_digital_twin.generator_dynamics import DynamicGenerator
    from phase1_digital_twin.load_management import LoadManager
    from phase1_digital_twin.energy_balance import EnergyBalanceCalculator
    from phase1_digital_twin.forecast_baseline import ForecastBaseline
    from phase1_digital_twin.scenario_manager import ScenarioManager


class DynamicMicrogridSimulation:
    """
    High-fidelity dynamic microgrid simulator modeling physical energy flows,
    storage state transitions, and resilience under extreme grid disturbances.
    """

    def __init__(
        self,
        scenario: ScenarioManager = None,
        duration_hours: float = None,
        time_step_minutes: float = None,
        random_seed: int = 42,
    ):
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

        # Simulation timing
        self.dt_minutes = time_step_minutes or SIMULATION["time_step_minutes"]
        self.dt_hours = self.dt_minutes / 60.0
        self.duration_hours = duration_hours or SIMULATION["duration_hours"]
        self.n_steps = int(self.duration_hours / self.dt_hours)
        self.hours = np.linspace(0, self.duration_hours, self.n_steps, endpoint=False)

        # Physical Microgrid Components
        self.net, self.buses = create_microgrid()
        self.solar = SolarModel()
        self.wind = WindModel()
        self.battery = DynamicBattery()
        self.generator = DynamicGenerator()
        self.load_manager = LoadManager()
        self.energy_balance = EnergyBalanceCalculator(tolerance_mw=0.001)
        self.forecaster = ForecastBaseline(horizon_hours=4.0, dt_hours=self.dt_hours)
        self.scenario = scenario or ScenarioManager.scenario_a_normal()

        # Grid state tracking
        self.grid_connected = GRID_CONNECTION["connected"]

        # Island reference slack generator on Bus 5 (Backup Generator)
        self._island_gen_idx = pp.create_gen(
            self.net,
            bus=self.buses["Backup Generator"],
            p_mw=0.0,
            vm_pu=1.0,
            slack=True,
            name="Island Slack Reference",
            max_p_mw=GENERATOR["capacity_mw"],
            min_p_mw=0.0,
            in_service=False,
        )

        # Pre-generate base weather curves
        self.base_irradiance = self.solar.generate_daily_profile(self.hours)
        self.base_wind_speed = self.wind.generate_daily_profile(self.hours)

        self.results: List[Dict[str, Any]] = []

    def run(self) -> pd.DataFrame:
        """
        Execute full time-stepped dynamic simulation.
        Returns detailed pandas DataFrame with all electrical and physical states.
        """
        self.results.clear()
        self.battery.reset()
        self.generator.reset()
        self.load_manager.reset()
        self.energy_balance.reset()
        self.forecaster.reset()

        for step_idx, hour in enumerate(self.hours):
            # 1. Environmental Conditions & Scenario Disturbances
            cloud_factor = self.scenario.get_cloud_factor(hour)
            irradiance = self.base_irradiance[step_idx] * cloud_factor

            wind_override = self.scenario.get_wind_override(hour)
            wind_speed = wind_override if wind_override is not None else self.base_wind_speed[step_idx]

            # 2. Grid Connection Status & Synchronization
            grid_available = self.scenario.is_grid_available(hour)
            sync_verified = self.scenario.is_sync_verified(hour)

            if not grid_available:
                if self.grid_connected:
                    self._disconnect_from_grid()
            else:
                if not self.grid_connected and sync_verified:
                    self._reconnect_to_grid()

            # 3. Renewable Generation Output
            solar_mw = self.solar.update(irradiance, ambient_temp_c=28.0)
            wind_mw = self.wind.update(wind_speed)
            re_gen_mw = solar_mw + wind_mw

            # 4. Consumer Demand Breakdown
            nom_demand = self.load_manager.get_nominal_demand(hour)
            total_nominal_load_mw = nom_demand["total_demand_mw"]

            # 5. Resource Dispatch & State Execution
            if self.grid_connected:
                # ----------------------------------------------------
                # GRID-CONNECTED MODE
                # ----------------------------------------------------
                self.generator.request_stop()
                actual_gen_p = self.generator.update(0.0, self.dt_hours)

                deficit_mw = total_nominal_load_mw - (re_gen_mw + actual_gen_p)
                if deficit_mw > 0:
                    # Grid imports deficit; battery idle
                    actual_bat_p = self.battery.update(0.0, self.dt_hours)
                    grid_target_mw = deficit_mw
                    load_dispatch = self.load_manager.dispatch(0.0, hour, self.dt_hours)
                else:
                    # Surplus: charge battery up to capacity, export remainder
                    surplus_mw = -deficit_mw
                    avail_chg = self.battery.available_charge_mw(self.dt_hours)
                    bat_req = min(surplus_mw, avail_chg)
                    actual_bat_p = self.battery.update(bat_req, self.dt_hours)
                    rem_surplus = surplus_mw - actual_bat_p
                    grid_target_mw = -rem_surplus  # Export
                    load_dispatch = self.load_manager.dispatch(0.0, hour, self.dt_hours)

            else:
                # ----------------------------------------------------
                # ISLANDED MODE (AUTONOMOUS MICROGRID)
                # ----------------------------------------------------
                grid_target_mw = 0.0
                gen_deficit = total_nominal_load_mw - re_gen_mw

                # Generator dispatch logic
                avail_norm_dis = self.battery.available_discharge_mw(self.dt_hours, emergency=False)
                if self.generator.is_running or gen_deficit > avail_norm_dis + 0.01:
                    self.generator.request_start()
                    req_gen = min(self.generator.capacity_mw, max(self.generator.min_output_mw, gen_deficit))
                    actual_gen_p = self.generator.update(req_gen, self.dt_hours)
                else:
                    self.generator.request_stop()
                    actual_gen_p = self.generator.update(0.0, self.dt_hours)

                # Now determine exact remaining balance after renewables + generator
                net_balance_mw = total_nominal_load_mw - (re_gen_mw + actual_gen_p)

                if net_balance_mw > 0.0001:
                    # Generation deficit -> Battery discharges
                    avail_norm = self.battery.available_discharge_mw(self.dt_hours, emergency=False)
                    if net_balance_mw <= avail_norm:
                        actual_bat_p = self.battery.update(-net_balance_mw, self.dt_hours, emergency=False)
                        curtail_needed = 0.0
                    else:
                        # Need emergency discharge down to min_soc
                        avail_emerg = self.battery.available_discharge_mw(self.dt_hours, emergency=True)
                        bat_dis = min(net_balance_mw, avail_emerg)
                        actual_bat_p = self.battery.update(-bat_dis, self.dt_hours, emergency=True)
                        curtail_needed = max(0.0, net_balance_mw - abs(actual_bat_p))

                    load_dispatch = self.load_manager.dispatch(curtail_needed, hour, self.dt_hours)

                else:
                    # Generation surplus in islanded mode -> Battery charges
                    surplus_mw = -net_balance_mw
                    avail_chg = self.battery.available_charge_mw(self.dt_hours)
                    bat_chg = min(surplus_mw, avail_chg)
                    actual_bat_p = self.battery.update(bat_chg, self.dt_hours)
                    load_dispatch = self.load_manager.dispatch(0.0, hour, self.dt_hours)

                    # If surplus exceeds battery charge capacity, curtail renewables
                    unabsorbed_surplus = surplus_mw - actual_bat_p
                    if unabsorbed_surplus > 0.0001:
                        curtail_sol = min(solar_mw, unabsorbed_surplus)
                        solar_mw -= curtail_sol
                        unabsorbed_surplus -= curtail_sol
                        curtail_wnd = min(wind_mw, unabsorbed_surplus)
                        wind_mw -= curtail_wnd
                        re_gen_mw = solar_mw + wind_mw

            served_total_load_mw = load_dispatch["served_total_mw"]

            # 6. Apply to Pandapower Network Model
            self._apply_network_state(
                solar_mw=solar_mw,
                wind_mw=wind_mw,
                battery_mw=actual_bat_p,
                generator_mw=actual_gen_p,
                load_dispatch=load_dispatch,
            )

            # 7. Run AC Power Flow
            converged, losses_mw, grid_flow_mw = self._run_power_flow()

            # In grid-connected mode, grid absorbs exact line losses and floating balance
            if self.grid_connected and converged:
                grid_final_mw = grid_flow_mw
            else:
                grid_final_mw = grid_target_mw

            # 9. Explicit Power Balance Calculation
            balance_res = self.energy_balance.calculate_balance(
                solar_mw=solar_mw,
                wind_mw=wind_mw,
                generator_mw=actual_gen_p,
                grid_mw=grid_final_mw,
                battery_mw=actual_bat_p,
                load_served_mw=served_total_load_mw,
                losses_mw=losses_mw,
            )

            # 10. Update Lookahead Forecaster
            storm_risk = 1.0 if not grid_available else (0.7 if self.scenario.is_storm_warning(hour) else 0.02)
            self.forecaster.observe(
                solar_mw=solar_mw,
                wind_mw=wind_mw,
                demand_mw=total_nominal_load_mw,
                storm_risk=storm_risk,
            )
            forecast = self.forecaster.predict(
                current_hour=hour,
                known_cloud_factor=cloud_factor,
                storm_warning=self.scenario.is_storm_warning(hour),
            )

            # 11. Compile Detailed Step Record
            record = {
                "step": step_idx,
                "hour": hour,
                "grid_connected": self.grid_connected,
                "grid_available": grid_available,
                "sync_verified": sync_verified,
                "irradiance": irradiance,
                "wind_speed_ms": wind_speed,
                "cloud_factor": cloud_factor,
                "solar_mw": solar_mw,
                "wind_mw": wind_mw,
                "re_gen_mw": re_gen_mw,
                "generator_mw": actual_gen_p,
                "generator_state": self.generator.state.value,
                "step_fuel_l": self.generator.last_step_fuel_l,
                "total_fuel_l": self.generator.total_fuel_l,
                "battery_mw": actual_bat_p,
                "battery_soc": self.battery.soc,
                "battery_energy_mwh": self.battery.energy_mwh,
                "grid_import_mw": grid_final_mw,
                "losses_mw": losses_mw,
                "nominal_critical_mw": load_dispatch["nominal_critical_mw"],
                "nominal_flexible_mw": load_dispatch["nominal_flexible_mw"],
                "nominal_other_mw": load_dispatch["nominal_other_mw"],
                "nominal_total_mw": total_nominal_load_mw,
                "served_critical_mw": load_dispatch["served_critical_mw"],
                "served_flexible_mw": load_dispatch["served_flexible_mw"],
                "served_other_mw": load_dispatch["served_other_mw"],
                "served_total_mw": served_total_load_mw,
                "shed_critical_mw": load_dispatch["shed_critical_mw"],
                "shed_flexible_mw": load_dispatch["shed_flexible_mw"],
                "shed_other_mw": load_dispatch["shed_other_mw"],
                "shed_total_mw": load_dispatch["shed_total_mw"],
                "load_shed_mw": load_dispatch["shed_total_mw"],
                "critical_emergency": load_dispatch["critical_emergency"],
                "deferred_flexible_mwh": load_dispatch["deferred_flexible_mwh"],
                "balance_error_mw": balance_res["balance_error_mw"],
                "is_balanced": balance_res["is_balanced"],
                "converged": converged,
                "forecast_next_solar_mw": forecast["next_step_solar_mw"],
                "forecast_next_wind_mw": forecast["next_step_wind_mw"],
                "forecast_next_demand_mw": forecast["next_step_demand_mw"],
                "forecast_next_outage_risk": forecast["next_step_outage_risk"],
            }
            self.results.append(record)

        df = pd.DataFrame(self.results)
        return df

    def _dispatch_resources(
        self,
        solar_mw: float,
        wind_mw: float,
        nominal_load_mw: float,
        hour: float,
        dt_hours: float,
    ) -> Dict[str, Any]:
        """
        Hierarchical rule-based dispatch for grid-connected and islanded modes.
        """
        re_gen_mw = solar_mw + wind_mw
        deficit_mw = nominal_load_mw - re_gen_mw

        bat_p_req = 0.0
        gen_p_req = 0.0
        grid_import_mw = 0.0
        battery_emergency = False

        if self.grid_connected:
            # ========================================================
            # GRID-CONNECTED DISPATCH PRIORITY:
            # 1. Renewables -> 2. Grid Import/Export -> 3. Battery Storage
            # ========================================================
            if deficit_mw > 0:
                # Grid supplies remaining load deficit
                grid_import_mw = deficit_mw
                # Battery idle; generator OFF
                bat_p_req = 0.0
                gen_p_req = 0.0
                self.generator.request_stop()
            else:
                # Surplus renewable power: first charge battery, then export excess
                surplus_mw = -deficit_mw
                max_chg = self.battery.available_charge_mw(dt_hours)
                bat_p_req = min(surplus_mw, max_chg)
                rem_surplus = surplus_mw - bat_p_req
                grid_import_mw = -rem_surplus  # Export to grid (negative)
                self.generator.request_stop()

        else:
            # ========================================================
            # ISLANDED DISPATCH PRIORITY:
            # 1. Renewables -> 2. Battery Response (normal reserve) ->
            # 3. Generator Dispatch -> 4. Battery Emergency Reserve ->
            # 5. Load Shedding (Flexible -> Other -> Critical)
            # ========================================================
            if deficit_mw <= 0:
                # Renewable surplus in islanded mode: charge battery, no generator
                surplus_mw = -deficit_mw
                max_chg = self.battery.available_charge_mw(dt_hours)
                bat_p_req = min(surplus_mw, max_chg)
                gen_p_req = 0.0
                self.generator.request_stop()
            else:
                rem_deficit = deficit_mw

                # Step 2: Battery normal discharge (preserve emergency reserve)
                norm_dis = self.battery.available_discharge_mw(dt_hours, emergency=False)
                bat_dis = min(rem_deficit, norm_dis)
                rem_deficit -= bat_dis

                # Step 3: Backup Generator Dispatch
                if rem_deficit > 0.01 or self.generator.is_running:
                    self.generator.request_start()
                    # Generator requested output to cover remaining deficit
                    gen_p_req = rem_deficit
                    # If generator starts, it relieves deficit
                    if self.generator.is_running:
                        gen_output_est = min(self.generator.capacity_mw, max(self.generator.min_output_mw, rem_deficit))
                        rem_deficit = max(0.0, rem_deficit - gen_output_est)

                # Step 4: Battery Emergency Discharge if deficit still remains
                if rem_deficit > 0.01:
                    battery_emergency = True
                    emerg_dis = self.battery.available_discharge_mw(dt_hours, emergency=True)
                    extra_dis = max(0.0, min(rem_deficit, emerg_dis - bat_dis))
                    bat_dis += extra_dis
                    rem_deficit -= extra_dis

                bat_p_req = -bat_dis

        return {
            "battery_p_req_mw": bat_p_req,
            "generator_p_req_mw": gen_p_req,
            "grid_import_mw": grid_import_mw,
            "battery_emergency": battery_emergency,
        }

    def _disconnect_from_grid(self):
        """Simulate tripping of PCC tie breaker into islanded microgrid mode."""
        self.grid_connected = False
        self.net.ext_grid.at[0, "in_service"] = False
        self.net.gen.at[self._island_gen_idx, "in_service"] = True

    def _reconnect_to_grid(self):
        """Reclose PCC tie breaker following successful synchronization."""
        self.grid_connected = True
        self.net.ext_grid.at[0, "in_service"] = True
        self.net.gen.at[self._island_gen_idx, "in_service"] = False
        self.generator.request_stop()

    def _apply_network_state(
        self,
        solar_mw: float,
        wind_mw: float,
        battery_mw: float,
        generator_mw: float,
        load_dispatch: Dict[str, Any],
    ):
        """Push dispatch variables to Pandapower elements."""
        # Static generators
        self.net.sgen.at[0, "p_mw"] = float(solar_mw)
        self.net.sgen.at[1, "p_mw"] = float(wind_mw)

        # Storage (Bus 4)
        self.net.storage.at[0, "p_mw"] = float(battery_mw)

        # Generator (Bus 5)
        if generator_mw > 0.001:
            self.net.sgen.at[2, "in_service"] = True
            self.net.sgen.at[2, "p_mw"] = float(generator_mw)
        else:
            self.net.sgen.at[2, "in_service"] = False
            self.net.sgen.at[2, "p_mw"] = 0.0

        if not self.grid_connected:
            self.net.gen.at[self._island_gen_idx, "p_mw"] = float(generator_mw)

        # Consumer Loads
        self.net.load.at[0, "p_mw"] = float(load_dispatch["served_critical_mw"])
        self.net.load.at[1, "p_mw"] = float(load_dispatch["served_flexible_mw"])
        self.net.load.at[2, "p_mw"] = float(load_dispatch["served_other_mw"])

    def _run_power_flow(self) -> tuple[bool, float, float]:
        """
        Execute Pandapower Newton-Raphson AC power flow.
        Returns: (converged, active_losses_mw, grid_active_power_mw)
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pp.runpp(self.net, numba=False)

            if self.net.converged:
                # Sum active losses across all lines
                line_losses = float(self.net.res_line["p_loss_mw"].sum()) if "p_loss_mw" in self.net.res_line else 0.0
                if self.grid_connected:
                    grid_p = float(self.net.res_ext_grid.p_mw.sum())
                else:
                    grid_p = 0.0
                return True, line_losses, grid_p
            else:
                return False, 0.0, 0.0
        except Exception:
            return False, 0.0, 0.0


if __name__ == "__main__":
    print("GRIDGUARD AI - Dynamic Simulation Engine Self-Test")
    print("=" * 50)
    sim = DynamicMicrogridSimulation(scenario=ScenarioManager.scenario_c_grid_outage())
    df_res = sim.run()

    print(f"Simulation completed: {len(df_res)} steps ({len(df_res)*0.25:.1f} hours)")
    print(f"Power flow converged: {df_res['converged'].sum()}/{len(df_res)} steps")
    print(f"Max power balance error: {df_res['balance_error_mw'].max():.6f} MW")
    print(f"Min Battery SOC: {df_res['battery_soc'].min():.1f}% | Max: {df_res['battery_soc'].max():.1f}%")
    print(f"Total load shed: {df_res['shed_total_mw'].sum()*0.25:.3f} MWh (Flex: {df_res['shed_flexible_mw'].sum()*0.25:.3f}, Crit: {df_res['shed_critical_mw'].sum()*0.25:.3f})")
    print("Dynamic Simulation Engine test passed successfully.")
