"""
sim — Simulation, visualization, and testing tools.

Modules:
    sim_harness         Full 6-DOF physics simulation (RealisticPhysics, BatteryModel,
                        WindField, FakePerception) with step-based and batch APIs.
    interactive_sim     Live matplotlib dashboard with operator and environment controls.
    visualize_sim       2D top-down playback of a simulation run.
    visualize_sim_3d    3D animated playback with optional GIF export.
    regression_runner   Load YAML scenarios from scenarios/, run them, assert metrics.
    param_sweep         Grid-search over parameter axes, output CSV/table of metrics.
"""
