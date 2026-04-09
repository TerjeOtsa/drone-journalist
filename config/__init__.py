"""
config — All tunable system parameters.

The single entry point is ``SystemConfig`` which aggregates:
    ShotParams, SafetyParams, MissionParams, StabilityParams,
    FlightInterfaceParams, SimulationParams.

Every constant has a default, a unit comment, and lives in one place.
No magic numbers in module code — import from here.
"""
