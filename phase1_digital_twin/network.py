# GRIDGUARD AI
# Phase 1: Electrical Network Construction
# ============================================================
# Builds a pandapower network representing a small industrial
# campus microgrid with 9 buses, a PCC to the utility grid,
# distributed generation, battery storage, a backup generator,
# and three categories of load.
# ============================================================

import pandapower as pp
from config import GRID, SOLAR, WIND, BATTERY, GENERATOR, GRID_CONNECTION, LOADS


def create_microgrid():
    """
    Construct the pandapower network for the GRIDGUARD AI microgrid.

    Bus layout:
        Bus 0 — Utility grid / PCC (slack bus)
        Bus 1 — Main distribution bus
        Bus 2 — Solar PV connection
        Bus 3 — Wind turbine connection
        Bus 4 — Battery storage connection
        Bus 5 — Backup generator connection
        Bus 6 — Critical loads
        Bus 7 — Flexible loads
        Bus 8 — Other loads

    Returns:
        net  : pandapower network
        buses: dict mapping bus names to bus indices
    """
    net = pp.create_empty_network(name="GRIDGUARD AI Microgrid")

    vn_kv = GRID["voltage_kv"]

    # ── Buses ───────────────────────────────────────────────
    buses = {}
    bus_defs = [
        ("Utility Grid / PCC",   vn_kv),
        ("Main Distribution",    vn_kv),
        ("Solar PV",             vn_kv),
        ("Wind Turbine",         vn_kv),
        ("Battery Storage",      vn_kv),
        ("Backup Generator",     vn_kv),
        ("Critical Loads",       vn_kv),
        ("Flexible Loads",       vn_kv),
        ("Other Loads",          vn_kv),
    ]
    for name, voltage in bus_defs:
        idx = pp.create_bus(net, vn_kv=voltage, name=name)
        buses[name] = idx

    # ── External grid (utility) at PCC ──────────────────────
    pp.create_ext_grid(
        net,
        bus=buses["Utility Grid / PCC"],
        vm_pu=1.0,
        va_degree=0,
        max_p_mw=GRID_CONNECTION["max_import_mw"],
        min_p_mw=-GRID_CONNECTION["max_import_mw"],
        name="Utility Grid",
        in_service=GRID_CONNECTION["connected"],
    )

    # ── Distribution lines (Bus 0 ↔ Bus 1, then Bus 1 ↔ each sub-bus) ─
    # Using typical 11 kV cable parameters for a short campus feeder.
    line_params = {
        "length_km":          0.5,
        "r_ohm_per_km":       0.32,
        "x_ohm_per_km":       0.07,
        "c_nf_per_km":        0,
        "max_i_ka":           0.2,
    }

    # PCC → Main distribution
    pp.create_line_from_parameters(
        net,
        from_bus=buses["Utility Grid / PCC"],
        to_bus=buses["Main Distribution"],
        name="PCC to Main",
        **line_params,
    )

    # Main distribution → every resource / load bus
    downstream = [
        "Solar PV", "Wind Turbine", "Battery Storage",
        "Backup Generator", "Critical Loads", "Flexible Loads", "Other Loads",
    ]
    for bus_name in downstream:
        pp.create_line_from_parameters(
            net,
            from_bus=buses["Main Distribution"],
            to_bus=buses[bus_name],
            length_km=0.2,
            r_ohm_per_km=0.32,
            x_ohm_per_km=0.07,
            c_nf_per_km=0,
            max_i_ka=0.2,
            name=f"Main to {bus_name}",
        )

    # ── Generators (static generators in pandapower) ────────
    # Solar PV
    pp.create_sgen(
        net,
        bus=buses["Solar PV"],
        p_mw=SOLAR["initial_output_mw"],
        q_mvar=0,
        name="Solar PV",
        type="PV",
        max_p_mw=SOLAR["capacity_mw"],
        min_p_mw=0,
    )

    # Wind turbine
    pp.create_sgen(
        net,
        bus=buses["Wind Turbine"],
        p_mw=WIND["initial_output_mw"],
        q_mvar=0,
        name="Wind Turbine",
        type="WP",
        max_p_mw=WIND["capacity_mw"],
        min_p_mw=0,
    )

    # ── Battery (modelled as a pandapower storage element) ──
    pp.create_storage(
        net,
        bus=buses["Battery Storage"],
        p_mw=0,                              # idle initially
        max_e_mwh=BATTERY["capacity_mwh"],
        soc_percent=BATTERY["initial_soc"],
        max_p_mw=BATTERY["max_power_mw"],
        min_p_mw=-BATTERY["max_power_mw"],   # discharge is negative
        q_mvar=0,
        name="Battery",
    )

    # ── Backup generator (static generator, initially OFF) ──
    pp.create_sgen(
        net,
        bus=buses["Backup Generator"],
        p_mw=GENERATOR["initial_output_mw"],
        q_mvar=0,
        name="Backup Generator",
        type="Diesel",
        max_p_mw=GENERATOR["capacity_mw"],
        min_p_mw=0,
        in_service=False,                     # OFF by default
    )

    # ── Loads ───────────────────────────────────────────────
    pp.create_load(
        net,
        bus=buses["Critical Loads"],
        p_mw=LOADS["critical_mw"],
        q_mvar=0.05,
        name="Critical Loads",
    )

    pp.create_load(
        net,
        bus=buses["Flexible Loads"],
        p_mw=LOADS["flexible_mw"],
        q_mvar=0.03,
        name="Flexible Loads",
    )

    pp.create_load(
        net,
        bus=buses["Other Loads"],
        p_mw=LOADS["other_mw"],
        q_mvar=0.04,
        name="Other Consumers",
    )

    return net, buses


# ────────────────────────────────────────────────────────────
# Quick self-test: run power flow and print diagnostics
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    net, buses = create_microgrid()

    # Run electrical power flow
    pp.runpp(net, numba=False)

    print("\nGRIDGUARD AI - MICROGRID INITIALIZATION")
    print("=" * 50)

    print("\nBus Information:")
    print(net.bus[["name", "vn_kv"]])

    print("\nGeneration (static generators):")
    print(net.sgen[["name", "p_mw", "in_service"]])

    print("\nStorage:")
    print(net.storage[["name", "p_mw", "soc_percent", "max_e_mwh"]])

    print("\nLoads:")
    print(net.load[["name", "p_mw"]])

    print("\nPower Flow Results:")
    print(f"  Utility Grid Supply:     {net.res_ext_grid.p_mw.sum(): .4f} MW")
    print(f"  Total Load:              {net.res_load.p_mw.sum(): .4f} MW")
    print(f"  Solar + Wind Generation: {net.res_sgen.p_mw.sum(): .4f} MW")

    print("\nBus Voltages (per unit and angle):")
    voltage_df = net.res_bus[["vm_pu", "va_degree"]].copy()
    voltage_df.index = net.bus["name"]
    print(voltage_df)

    print("\nLine Loading (%):")
    line_df = net.res_line[["loading_percent"]].copy()
    line_df.index = net.line["name"]
    print(line_df)

    print(f"\nConverged: {net.converged}")
