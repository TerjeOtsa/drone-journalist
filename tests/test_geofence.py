"""
Unit tests for polygon-based geofence geometry and integration with SafetyModule.
"""

import math

from config.parameters import SafetyParams
from flight.geofence import (
    GeofenceChecker,
    GeofenceConfig,
    GeofenceZone,
    circle_polygon,
    distance_to_polygon_edge,
    point_in_polygon,
)
from flight.safety_module import SafetyModule
from interfaces.schemas import DroneTelemetry, FlightSetpoint, SafetyOverride, Vec3

# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

SQUARE_100 = [(50, 50), (50, -50), (-50, -50), (-50, 50)]  # 100×100 square

CONCAVE_L = [  # L-shape (concave)
    (0, 0), (0, 40), (20, 40), (20, 20), (40, 20), (40, 0),
]

def _telem(x=0.0, y=0.0, alt=5.0, in_air=True, batt=80) -> DroneTelemetry:
    return DroneTelemetry(
        position=Vec3(x, y, -alt),  # NED: z is negative for altitude
        in_air=in_air,
        battery_percent=batt,
        home_position=Vec3(0, 0, 0),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Point-in-polygon tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPointInPolygon:
    def test_origin_inside_square(self):
        assert point_in_polygon(0, 0, SQUARE_100) is True

    def test_outside_square(self):
        assert point_in_polygon(60, 0, SQUARE_100) is False

    def test_far_outside(self):
        assert point_in_polygon(200, 200, SQUARE_100) is False

    def test_inside_concave_L_shape(self):
        # Inside the bottom-left part of the L
        assert point_in_polygon(10, 10, CONCAVE_L) is True

    def test_outside_concave_notch(self):
        # The indentation of the L — (30, 30) is outside
        assert point_in_polygon(30, 30, CONCAVE_L) is False

    def test_inside_upper_arm_of_L(self):
        # (10, 30) is inside the upper arm
        assert point_in_polygon(10, 30, CONCAVE_L) is True

    def test_degenerate_polygon_too_few_vertices(self):
        assert point_in_polygon(0, 0, [(0, 0), (1, 1)]) is False

    def test_triangle(self):
        tri = [(0, 0), (10, 0), (5, 10)]
        assert point_in_polygon(5, 3, tri) is True
        assert point_in_polygon(0, 10, tri) is False


class TestDistanceToPolygonEdge:
    def test_origin_to_square_edge(self):
        # Origin to nearest edge of 100×100 square centred at origin = 50m
        dist = distance_to_polygon_edge(0, 0, SQUARE_100)
        assert abs(dist - 50.0) < 0.01

    def test_close_to_edge(self):
        dist = distance_to_polygon_edge(49, 0, SQUARE_100)
        assert abs(dist - 1.0) < 0.01

    def test_outside_distance(self):
        dist = distance_to_polygon_edge(55, 0, SQUARE_100)
        assert abs(dist - 5.0) < 0.01

    def test_corner_distance(self):
        # Distance from (60, 60) to corner (50, 50)
        dist = distance_to_polygon_edge(60, 60, SQUARE_100)
        expected = math.hypot(10, 10)
        assert abs(dist - expected) < 0.01


class TestCirclePolygon:
    def test_correct_vertex_count(self):
        poly = circle_polygon(50.0, n_vertices=36)
        assert len(poly) == 36

    def test_vertices_on_circle(self):
        r = 75.0
        poly = circle_polygon(r, 24)
        for x, y in poly:
            assert abs(math.hypot(x, y) - r) < 1e-9

    def test_origin_inside(self):
        poly = circle_polygon(100.0)
        assert point_in_polygon(0, 0, poly) is True

    def test_outside_circle(self):
        poly = circle_polygon(100.0, 72)
        assert point_in_polygon(101, 0, poly) is False


# ═══════════════════════════════════════════════════════════════════════════
#  GeofenceChecker tests
# ═══════════════════════════════════════════════════════════════════════════

class TestGeofenceChecker:
    def test_inside_keep_in_ok(self):
        cfg = GeofenceConfig(
            keep_in=GeofenceZone(SQUARE_100),
            ceiling_m=50.0, floor_m=0.5,
        )
        gc = GeofenceChecker(cfg)
        r = gc.check(Vec3(10, 10, -5))  # alt=5m
        assert r.ok is True

    def test_outside_keep_in(self):
        cfg = GeofenceConfig(
            keep_in=GeofenceZone(SQUARE_100),
            ceiling_m=50.0, floor_m=0.5,
        )
        gc = GeofenceChecker(cfg)
        r = gc.check(Vec3(60, 0, -5))
        assert r.inside_keep_in is False
        assert r.ok is False

    def test_exclusion_zone_breach(self):
        excl = GeofenceZone(
            vertices=[(10, 10), (10, -10), (-10, -10), (-10, 10)],
            label="no-fly zone",
        )
        cfg = GeofenceConfig(
            keep_in=GeofenceZone(SQUARE_100),
            exclusion_zones=[excl],
            ceiling_m=50.0, floor_m=0.5,
        )
        gc = GeofenceChecker(cfg)
        # Inside exclusion zone
        r = gc.check(Vec3(0, 0, -5))
        assert r.violated_exclusion == "no-fly zone"
        assert r.ok is False

    def test_exclusion_zone_outside_is_ok(self):
        excl = GeofenceZone(
            vertices=[(10, 10), (10, -10), (-10, -10), (-10, 10)],
            label="no-fly zone",
        )
        cfg = GeofenceConfig(
            keep_in=GeofenceZone(SQUARE_100),
            exclusion_zones=[excl],
            ceiling_m=50.0, floor_m=0.5,
        )
        gc = GeofenceChecker(cfg)
        r = gc.check(Vec3(30, 30, -5))
        assert r.violated_exclusion is None
        assert r.ok is True

    def test_above_ceiling(self):
        cfg = GeofenceConfig(ceiling_m=30.0, floor_m=0.5)
        gc = GeofenceChecker(cfg, fallback_radius=100)
        r = gc.check(Vec3(0, 0, -35))  # alt=35 > ceiling 30
        assert r.altitude_ok is False
        assert r.ok is False

    def test_below_floor(self):
        cfg = GeofenceConfig(ceiling_m=30.0, floor_m=0.5)
        gc = GeofenceChecker(cfg, fallback_radius=100)
        r = gc.check(Vec3(0, 0, -0.2))  # alt=0.2 < floor 0.5
        assert r.altitude_ok is False

    def test_warn_margin_near_keep_in_edge(self):
        cfg = GeofenceConfig(
            keep_in=GeofenceZone(SQUARE_100),
            ceiling_m=50.0, floor_m=0.5,
            warn_margin_m=5.0,
        )
        gc = GeofenceChecker(cfg)
        # 3m from edge: inside warn margin
        r = gc.check(Vec3(47, 0, -10))
        assert r.ok is True  # still inside
        assert r.keep_in_margin < 5.0  # but within warn margin
        assert r.warn is True

    def test_no_warn_when_well_inside(self):
        cfg = GeofenceConfig(
            keep_in=GeofenceZone(SQUARE_100),
            ceiling_m=50.0, floor_m=0.5,
            warn_margin_m=5.0,
        )
        gc = GeofenceChecker(cfg)
        r = gc.check(Vec3(0, 0, -10))
        assert r.ok is True
        assert r.keep_in_margin == float("inf")

    def test_warn_margin_near_exclusion_zone(self):
        excl = GeofenceZone(
            vertices=[(10, 10), (10, -10), (-10, -10), (-10, 10)],
            label="building",
        )
        cfg = GeofenceConfig(
            keep_in=GeofenceZone(SQUARE_100),
            exclusion_zones=[excl],
            ceiling_m=50.0, floor_m=0.5,
            warn_margin_m=5.0,
        )
        gc = GeofenceChecker(cfg)
        # 3m from exclusion boundary, outside it
        r = gc.check(Vec3(13, 0, -10))
        assert r.violated_exclusion is None
        assert r.exclusion_margin < 5.0

    def test_altitude_warn_margin(self):
        cfg = GeofenceConfig(ceiling_m=30.0, floor_m=0.5, warn_margin_m=5.0)
        gc = GeofenceChecker(cfg, fallback_radius=100)
        r = gc.check(Vec3(0, 0, -28))  # alt=28, margin to ceiling=2 < 5
        assert r.altitude_ok is True
        assert r.altitude_margin < 5.0

    def test_fallback_cylinder_radius(self):
        """When no keep_in polygon, circle_polygon(fallback_radius) is used."""
        cfg = GeofenceConfig(ceiling_m=50.0, floor_m=0.5)
        gc = GeofenceChecker(cfg, fallback_radius=80)
        # Inside circle of radius 80
        r = gc.check(Vec3(50, 0, -5))
        assert r.inside_keep_in is True
        # Outside circle of radius 80
        r2 = gc.check(Vec3(85, 0, -5))
        assert r2.inside_keep_in is False

    def test_multiple_exclusion_zones(self):
        zones = [
            GeofenceZone([(5, 5), (5, -5), (-5, -5), (-5, 5)], "zone-A"),
            GeofenceZone([(35, 35), (35, 25), (25, 25), (25, 35)], "zone-B"),
        ]
        cfg = GeofenceConfig(
            keep_in=GeofenceZone(SQUARE_100),
            exclusion_zones=zones,
            ceiling_m=50.0, floor_m=0.5,
        )
        gc = GeofenceChecker(cfg)
        # In zone-A
        r = gc.check(Vec3(0, 0, -5))
        assert r.violated_exclusion == "zone-A"
        # In zone-B
        r2 = gc.check(Vec3(30, 30, -5))
        assert r2.violated_exclusion == "zone-B"
        # In neither
        r3 = gc.check(Vec3(20, 0, -5))
        assert r3.violated_exclusion is None
        assert r3.ok is True


# ═══════════════════════════════════════════════════════════════════════════
#  SafetyModule integration with polygon geofence
# ═══════════════════════════════════════════════════════════════════════════

class TestSafetyModulePolygonGeofence:
    """Verify that SafetyModule correctly delegates to GeofenceChecker."""

    def _safety_with_polygon(self, keep_in=None, exclusions=None,
                              ceiling=50.0, floor=0.5, warn_margin=5.0):
        cfg = GeofenceConfig(
            keep_in=keep_in,
            exclusion_zones=exclusions or [],
            ceiling_m=ceiling,
            floor_m=floor,
            warn_margin_m=warn_margin,
        )
        params = SafetyParams(geofence_config=cfg)
        sm = SafetyModule(params)
        sm.heartbeat()
        return sm

    def test_inside_polygon_ok(self):
        sm = self._safety_with_polygon(keep_in=GeofenceZone(SQUARE_100))
        st = sm.update(_telem(x=10, y=10, alt=5), FlightSetpoint())
        assert st.geofence_ok is True
        assert st.active_override == SafetyOverride.NONE

    def test_outside_polygon_returns_home(self):
        sm = self._safety_with_polygon(keep_in=GeofenceZone(SQUARE_100))
        st = sm.update(_telem(x=60, y=0, alt=5), FlightSetpoint())
        assert st.geofence_ok is False
        assert st.active_override == SafetyOverride.RETURN_HOME
        assert "keep-in" in st.reasons[0]

    def test_exclusion_zone_returns_home(self):
        excl = GeofenceZone(
            vertices=[(10, 10), (10, -10), (-10, -10), (-10, 10)],
            label="crowd",
        )
        sm = self._safety_with_polygon(
            keep_in=GeofenceZone(SQUARE_100),
            exclusions=[excl],
        )
        st = sm.update(_telem(x=0, y=0, alt=5), FlightSetpoint())
        assert st.geofence_ok is False
        assert st.active_override == SafetyOverride.RETURN_HOME
        assert "crowd" in st.reasons[0]

    def test_near_keep_in_edge_reduces_speed(self):
        sm = self._safety_with_polygon(
            keep_in=GeofenceZone(SQUARE_100),
            warn_margin=8.0,
        )
        # 3m from edge (at x=47, edge at x=50)
        st = sm.update(_telem(x=47, y=0, alt=10), FlightSetpoint())
        assert st.geofence_ok is False
        assert st.active_override == SafetyOverride.REDUCE_SPEED
        assert "keep-in" in st.reasons[0]

    def test_near_exclusion_reduces_speed(self):
        excl = GeofenceZone(
            vertices=[(10, 10), (10, -10), (-10, -10), (-10, 10)],
            label="building",
        )
        sm = self._safety_with_polygon(
            keep_in=GeofenceZone(SQUARE_100),
            exclusions=[excl],
            warn_margin=5.0,
        )
        # 3m from exclusion boundary, outside exclusion
        st = sm.update(_telem(x=13, y=0, alt=10), FlightSetpoint())
        assert st.geofence_ok is False
        assert st.active_override == SafetyOverride.REDUCE_SPEED
        assert "exclusion" in st.reasons[0]

    def test_ceiling_breach_with_polygon(self):
        sm = self._safety_with_polygon(
            keep_in=GeofenceZone(SQUARE_100),
            ceiling=20.0,
        )
        st = sm.update(_telem(x=0, y=0, alt=25), FlightSetpoint())
        assert st.geofence_ok is False
        assert st.active_override == SafetyOverride.RETURN_HOME

    def test_below_floor_in_air_hovers(self):
        sm = self._safety_with_polygon(
            keep_in=GeofenceZone(SQUARE_100),
            floor=2.0,
        )
        st = sm.update(_telem(x=0, y=0, alt=1.0, in_air=True), FlightSetpoint())
        assert st.geofence_ok is False
        assert st.active_override == SafetyOverride.HOVER

    def test_legacy_cylinder_fallback(self):
        """No geofence_config → cylinder from SafetyParams.geofence_radius."""
        sm = SafetyModule(SafetyParams(geofence_radius=80))
        sm.heartbeat()
        # Inside radius 80
        st = sm.update(_telem(x=50, y=0, alt=5), FlightSetpoint())
        assert st.geofence_ok is True
        # Outside radius 80
        st2 = sm.update(_telem(x=85, y=0, alt=5), FlightSetpoint())
        assert st2.geofence_ok is False
        assert st2.active_override == SafetyOverride.RETURN_HOME
