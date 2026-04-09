"""
Product-side recording and preview profiles.

These profiles intentionally stay in the product domain. They do not mutate
flight-control tuning; they define the behavior of the app/video/logging slice.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProductProfile:
    """Named field profile used by the app/video layer."""

    name: str
    pre_roll_sec: int
    preview_max_bitrate_kbps: int
    record_ready_hold_ms: int
    record_stop_hold_ms: int
    clip_segment_sec: int = 10
    preview_width: int = 1280
    preview_height: int = 720


@dataclass(frozen=True)
class RecordingPolicy:
    """Recorder timing parameters derived from the active profile."""

    pre_roll_sec: int
    ready_hold_s: float
    stop_hold_s: float
    clip_segment_sec: int


@dataclass(frozen=True)
class ProductRuntimeConfig:
    """Resolved product configuration used by app/video services."""

    active_profile: ProductProfile
    session_root: str = "/data/sessions"
    auto_download_in_flight: bool = False
    metadata_sample_hz: float = 2.0
    preview_low_bitrate_kbps: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "preview_low_bitrate_kbps",
            max(400, self.active_profile.preview_max_bitrate_kbps // 3),
        )


_PROFILES: dict[str, ProductProfile] = {
    "stand_up": ProductProfile(
        name="stand_up",
        pre_roll_sec=8,
        preview_max_bitrate_kbps=1500,
        record_ready_hold_ms=300,
        record_stop_hold_ms=3000,
    ),
    "walk": ProductProfile(
        name="walk",
        pre_roll_sec=8,
        preview_max_bitrate_kbps=2500,
        record_ready_hold_ms=300,
        record_stop_hold_ms=2000,
    ),
    "safe": ProductProfile(
        name="safe",
        pre_roll_sec=4,
        preview_max_bitrate_kbps=1200,
        record_ready_hold_ms=500,
        record_stop_hold_ms=3500,
    ),
}


def list_profiles() -> list[str]:
    """Return sorted names of all registered product profiles."""
    return sorted(_PROFILES)


def get_profile(name: str) -> ProductProfile:
    """Look up a profile by *name*, raising ``ValueError`` if unknown."""
    try:
        return _PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown product profile: {name}") from exc


def default_profile() -> ProductProfile:
    """Return the default 'walk' profile."""
    return get_profile("walk")


def recording_policy(profile_name: str) -> RecordingPolicy:
    """Derive recording timing policy from the named profile."""
    profile = get_profile(profile_name)
    return RecordingPolicy(
        pre_roll_sec=profile.pre_roll_sec,
        ready_hold_s=profile.record_ready_hold_ms / 1000.0,
        stop_hold_s=profile.record_stop_hold_ms / 1000.0,
        clip_segment_sec=profile.clip_segment_sec,
    )


def build_runtime_config(profile_name: str = "walk") -> ProductRuntimeConfig:
    """Build a fully-resolved runtime config from the named profile."""
    return ProductRuntimeConfig(active_profile=get_profile(profile_name))
