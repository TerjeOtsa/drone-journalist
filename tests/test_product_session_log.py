"""
Tests for product-side session logging.
"""

import json
import sqlite3
import time

import pytest

from interfaces.schemas import (
    LockState,
    MissionState,
    MissionStatus,
    SafetyOverride,
    SafetyStatus,
    ShotMode,
    StabilityLevel,
    TargetTrack,
    Vec3,
)
from product.adapters import EventMirror, ProductStateAdapter
from product.schemas import ProductEvent, RecordingState
from product.session_log import SessionLog


def _mission() -> MissionStatus:
    return MissionStatus(
        timestamp=time.time(),
        state=MissionState.FILM,
        shot_mode=ShotMode.WALK_AND_TALK,
        ready_to_record=True,
        stability=StabilityLevel.NOMINAL,
        safety=SafetyStatus(active_override=SafetyOverride.NONE),
        message="film",
    )


def _track() -> TargetTrack:
    return TargetTrack(
        timestamp=time.time(),
        lock_state=LockState.LOCKED,
        identity_confidence=0.92,
        tracking_confidence=0.88,
        target_position_world=Vec3(5.0, 0.2, 0.0),
        target_velocity_world=Vec3(0.9, 0.0, 0.0),
    )


class TestSessionLog:
    def test_records_events_and_snapshots(self, tmp_path):
        adapter = ProductStateAdapter(drone_id="drn_test", profile="walk")
        mirror = EventMirror()

        with SessionLog(tmp_path, "sess_001") as log:
            event = mirror.mirror_system_event(
                event=type("Event", (), {
                    "timestamp": 10.0,
                    "source": "mission",
                    "event": "state_change",
                    "payload": {"old": "lock", "new": "film"},
                })(),
                session_id="sess_001",
                monotonic_time=4.0,
            )
            snapshot = adapter.make_snapshot(
                "sess_001",
                _mission(),
                _track(),
                recording_state=RecordingState.RECORDING_CONFIRMED,
                active_clip_id="clip_001",
            )

            log.record_event(event)
            log.record_snapshot(snapshot)

            events = list(log.iter_events())
            snapshots = list(log.iter_snapshots())
            timeline = log.timeline()

        conn = sqlite3.connect(tmp_path / "sess_001" / "session.sqlite")
        try:
            conn.row_factory = sqlite3.Row
            meta = {
                row["key"]: row["value"]
                for row in conn.execute("SELECT key, value FROM meta")
            }
            user_version = conn.execute("PRAGMA user_version").fetchone()[0]
            journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        finally:
            conn.close()

        assert len(events) == 1
        assert events[0].seq == 1
        assert events[0].payload["new"] == "film"

        assert len(snapshots) == 1
        assert snapshots[0]["mission"]["mission_state"] == "film"
        assert snapshots[0]["recording_state"] == "recording_confirmed"
        assert snapshots[0]["active_clip_id"] == "clip_001"

        assert [item["kind"] for item in timeline] == ["event", "snapshot"]
        assert meta["session_id"] == "sess_001"
        assert meta["schema_version"] == "1"
        assert user_version == 1
        assert journal_mode.lower() == "wal"
        assert (tmp_path / "sess_001" / "session.sqlite").exists()
        assert (tmp_path / "sess_001" / "events.jsonl").exists()

    def test_duplicate_event_sequence_is_rejected(self, tmp_path):
        mirror = EventMirror()
        first = mirror.mirror_system_event(
            event=type("Event", (), {
                "timestamp": 10.0,
                "source": "mission",
                "event": "state_change",
                "payload": {"old": "lock", "new": "film"},
            })(),
            session_id="sess_dup",
            monotonic_time=4.0,
        )
        duplicate = ProductEvent(
            seq=first.seq,
            timestamp=11.0,
            monotonic_time=5.0,
            session_id="sess_dup",
            source="mission",
            event="state_change",
            payload={"old": "film", "new": "degrade"},
        )

        with SessionLog(tmp_path, "sess_dup") as log:
            log.record_event(first)
            with pytest.raises(sqlite3.IntegrityError):
                log.record_event(duplicate)

            events = list(log.iter_events())

        assert len(events) == 1
        assert events[0].payload["new"] == "film"
        with open(tmp_path / "sess_dup" / "events.jsonl", "r", encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle if line.strip()]
        assert len(lines) == 1
        assert lines[0]["payload"]["new"] == "film"

    def test_jsonl_failure_rolls_back_event_write(self, tmp_path, monkeypatch):
        mirror = EventMirror()
        event = mirror.mirror_system_event(
            event=type("Event", (), {
                "timestamp": 10.0,
                "source": "mission",
                "event": "state_change",
                "payload": {"old": "lock", "new": "film"},
            })(),
            session_id="sess_fail",
            monotonic_time=4.0,
        )

        with SessionLog(tmp_path, "sess_fail") as log:
            def _fail(*_args, **_kwargs):
                raise RuntimeError("disk full")

            monkeypatch.setattr(log, "_append_jsonl_line", _fail)

            with pytest.raises(RuntimeError, match="disk full"):
                log.record_event(event)

            events = list(log.iter_events())

        assert events == []
        assert not (tmp_path / "sess_fail" / "events.jsonl").exists()
