"""
Tests for product-side profile selection.
"""

import pytest

from product.config_profiles import (
    build_runtime_config,
    default_profile,
    get_profile,
    list_profiles,
    recording_policy,
)


class TestProfiles:
    def test_profiles_are_listed_in_stable_order(self):
        assert list_profiles() == ["safe", "stand_up", "walk"]

    def test_default_profile_is_walk(self):
        profile = default_profile()

        assert profile.name == "walk"
        assert profile.pre_roll_sec == 8
        assert profile.preview_max_bitrate_kbps == 2500

    def test_recording_policy_uses_profile_timing(self):
        policy = recording_policy("safe")

        assert policy.pre_roll_sec == 4
        assert policy.ready_hold_s == 0.5
        assert policy.stop_hold_s == 3.5
        assert policy.clip_segment_sec == 10

    def test_runtime_config_derives_low_bitrate_floor(self):
        runtime = build_runtime_config("stand_up")

        assert runtime.active_profile.name == "stand_up"
        assert runtime.preview_low_bitrate_kbps == 500
        assert runtime.auto_download_in_flight is False

    def test_unknown_profile_raises_value_error(self):
        with pytest.raises(ValueError):
            get_profile("crowd")
