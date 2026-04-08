"""
Product-side integration modules for video, app, logging, and configuration.
"""

from product.adapters import (
    EventMirror,
    LaunchTranslation,
    ProductCommandAdapter,
    ProductStateAdapter,
)
from product.operator_panel import (
    DISTANCE_PRESETS,
    OperatorControlPanel,
    OperatorPanelController,
    OperatorPanelViewState,
)
from product.config_profiles import (
    ProductProfile,
    ProductRuntimeConfig,
    RecordingPolicy,
    build_runtime_config,
    default_profile,
    get_profile,
    list_profiles,
    recording_policy,
)
from product.schemas import (
    ConfidenceSnapshot,
    LinkStats,
    LockStatusView,
    MissionView,
    ProductEvent,
    ProductSnapshot,
    RecordingState,
)
from product.session_log import SessionLog

__all__ = [
    "ConfidenceSnapshot",
    "EventMirror",
    "LaunchTranslation",
    "LinkStats",
    "LockStatusView",
    "MissionView",
    "DISTANCE_PRESETS",
    "OperatorControlPanel",
    "OperatorPanelController",
    "OperatorPanelViewState",
    "ProductCommandAdapter",
    "ProductEvent",
    "ProductProfile",
    "ProductRuntimeConfig",
    "ProductSnapshot",
    "ProductStateAdapter",
    "RecordingPolicy",
    "RecordingState",
    "SessionLog",
    "build_runtime_config",
    "default_profile",
    "get_profile",
    "list_profiles",
    "recording_policy",
]
