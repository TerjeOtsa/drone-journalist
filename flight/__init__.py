"""
flight — Core autonomy and control modules.

Modules:
    mission_state_machine   IDLE → TAKEOFF → ACQUIRE → LOCK → FILM → DEGRADE → RETURN → LAND
    shot_controller         Computes desired drone position for 4 shot modes + face-aware bias
    stability_supervisor    Assesses wind/jitter → speed scale + extra standoff distance
    safety_module           50 Hz geofence, battery, link-loss, proximity, emergency checks
    geofence                Polygon keep-in zones, exclusion zones, cylinder fallback
    flight_interface        MAVLink setpoint throttling, watchdog, arm/disarm
    pymavlink_transport     Serial/UDP MAVLink transport (hardware only)
"""
