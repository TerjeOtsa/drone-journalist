"""
Geofence Geometry
=================
Point-in-polygon keep-in zone plus exclusion zones, with altitude limits.

Supports:
  - **Keep-in polygon** — convex or concave, defined as a list of (north, east)
    vertices in NED metres relative to home.  The drone must stay inside.
  - **Exclusion zones** — one or more polygons the drone must stay outside.
  - **Altitude band** — [floor, ceiling] in metres AGL.
  - **Warning margin** — distance inside the keep-in boundary (or outside an
    exclusion boundary) at which a soft warning is raised.

The cylinder from the original SafetyParams is preserved as a default: if no
polygon vertices are supplied, the module constructs a circular approximation
using `geofence_radius`.

All geometry is 2-D horizontal (NED north/east).  Altitude is checked
separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from interfaces.schemas import Vec3

# Type alias: a polygon is a list of (north, east) vertices
Polygon = List[Tuple[float, float]]


@dataclass
class GeofenceZone:
    """A single keep-in or exclusion polygon with an optional label."""
    vertices: Polygon
    label: str = ""


@dataclass
class GeofenceConfig:
    """
    Full geofence specification.

    If `keep_in` is None the safety module will auto-generate a circular
    keep-in from the legacy `geofence_radius` parameter.
    """
    keep_in: Optional[GeofenceZone] = None
    exclusion_zones: List[GeofenceZone] = field(default_factory=list)
    ceiling_m: float = 30.0
    floor_m: float = 0.5
    warn_margin_m: float = 10.0


# ════════════════════════════════════════════════════════════════════════════
#  Polygon utilities
# ════════════════════════════════════════════════════════════════════════════

def point_in_polygon(px: float, py: float, poly: Polygon) -> bool:
    """
    Ray-casting algorithm for point-in-polygon.

    Works for convex and concave polygons.  The polygon is implicitly closed
    (last vertex connects to first).
    """
    n = len(poly)
    if n < 3:
        return False
    inside = False
    x1, y1 = poly[0]
    for i in range(1, n + 1):
        x2, y2 = poly[i % n]
        if py > min(y1, y2):
            if py <= max(y1, y2):
                if px <= max(x1, x2):
                    if y1 != y2:
                        xinters = (py - y1) * (x2 - x1) / (y2 - y1) + x1
                    if y1 == y2 or px <= xinters:
                        inside = not inside
        x1, y1 = x2, y2
    return inside


def distance_to_polygon_edge(px: float, py: float, poly: Polygon) -> float:
    """
    Minimum distance from point (px, py) to any edge of the polygon.
    """
    n = len(poly)
    if n < 2:
        return float("inf")
    min_dist = float("inf")
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        dist = _point_segment_distance(px, py, x1, y1, x2, y2)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def _point_segment_distance(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    """Distance from point P to segment AB."""
    abx = bx - ax
    aby = by - ay
    ab_sq = abx * abx + aby * aby
    if ab_sq < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab_sq))
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    return math.hypot(px - proj_x, py - proj_y)


def circle_polygon(radius: float, n_vertices: int = 36) -> Polygon:
    """
    Generate a regular polygon approximating a circle of `radius` metres
    centred at the origin.
    """
    return [
        (radius * math.cos(2 * math.pi * i / n_vertices),
         radius * math.sin(2 * math.pi * i / n_vertices))
        for i in range(n_vertices)
    ]


# ════════════════════════════════════════════════════════════════════════════
#  Geofence checker
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class GeofenceResult:
    """Result of a single geofence evaluation."""
    inside_keep_in: bool = True
    keep_in_margin: float = float("inf")   # distance to keep-in boundary
    violated_exclusion: Optional[str] = None  # label of violated zone, if any
    exclusion_margin: float = float("inf")  # distance to nearest exclusion edge
    altitude_ok: bool = True
    altitude_margin: float = float("inf")

    @property
    def ok(self) -> bool:
        return self.inside_keep_in and self.violated_exclusion is None and self.altitude_ok

    @property
    def warn(self) -> bool:
        """True if within the warning margin of any boundary."""
        return (
            self.keep_in_margin < float("inf")
            or self.exclusion_margin < float("inf")
            or self.altitude_margin < float("inf")
        )


class GeofenceChecker:
    """
    Stateless evaluator that checks a position against the full geofence config.

    Constructed once; call `check()` every tick.
    """

    def __init__(
        self,
        config: GeofenceConfig,
        fallback_radius: float = 100.0,
    ) -> None:
        self.cfg = config
        if config.keep_in is not None:
            self._keep_in = config.keep_in.vertices
        else:
            self._keep_in = circle_polygon(fallback_radius)
        self._exclusions = config.exclusion_zones

    def check(self, pos: Vec3, warn_margin: float | None = None) -> GeofenceResult:
        """
        Evaluate position against the geofence.

        Parameters
        ----------
        pos : Vec3
            NED position (metres from home).
        warn_margin : float, optional
            Override for the warning margin.  Defaults to config value.
        """
        wm = warn_margin if warn_margin is not None else self.cfg.warn_margin_m
        north, east = pos.x, pos.y
        alt = -pos.z  # NED: up is negative z

        # ── Keep-in ──────────────────────────────────────────────────────
        inside = point_in_polygon(north, east, self._keep_in)
        keep_in_dist = distance_to_polygon_edge(north, east, self._keep_in)
        keep_in_margin = keep_in_dist if inside and keep_in_dist < wm else float("inf")

        # ── Exclusion zones ──────────────────────────────────────────────
        violated_label: Optional[str] = None
        nearest_excl_margin = float("inf")
        for zone in self._exclusions:
            in_excl = point_in_polygon(north, east, zone.vertices)
            if in_excl:
                violated_label = zone.label or "exclusion"
                break
            excl_dist = distance_to_polygon_edge(north, east, zone.vertices)
            if excl_dist < wm:
                nearest_excl_margin = min(nearest_excl_margin, excl_dist)

        # ── Altitude ─────────────────────────────────────────────────────
        alt_ok = self.cfg.floor_m <= alt <= self.cfg.ceiling_m
        alt_margin = float("inf")
        if alt_ok:
            margin_to_ceil = self.cfg.ceiling_m - alt
            margin_to_floor = alt - self.cfg.floor_m
            smallest = min(margin_to_ceil, margin_to_floor)
            if smallest < wm:
                alt_margin = smallest

        return GeofenceResult(
            inside_keep_in=inside,
            keep_in_margin=keep_in_margin,
            violated_exclusion=violated_label,
            exclusion_margin=nearest_excl_margin,
            altitude_ok=alt_ok,
            altitude_margin=alt_margin,
        )
