"""
Prototype session logging for the product layer.

The long-term design calls for SQLite plus a raw binary event stream. The
current Python prototype implements SQLite plus JSONL so we can persist and
replay product events today without introducing protobuf infrastructure yet.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterator

from product.schemas import ProductEvent, ProductSnapshot


class SessionLog:
    """Durable session bundle for product events and snapshots."""

    SCHEMA_VERSION = 1
    SQLITE_BUSY_TIMEOUT_MS = 5000

    def __init__(self, root_dir: str | Path, session_id: str) -> None:
        self.session_id = session_id
        self.root_dir = Path(root_dir)
        self.session_dir = self.root_dir / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.session_dir / "session.sqlite"
        self.events_path = self.session_dir / "events.jsonl"

        self._conn = sqlite3.connect(self.db_path, timeout=self.SQLITE_BUSY_TIMEOUT_MS / 1000.0)
        self._conn.row_factory = sqlite3.Row
        self._apply_pragmas()
        self._initialise()

    def _apply_pragmas(self) -> None:
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute(f"PRAGMA busy_timeout = {self.SQLITE_BUSY_TIMEOUT_MS}")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")

    def _initialise(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY,
                timestamp REAL NOT NULL,
                monotonic_time REAL NOT NULL,
                source TEXT NOT NULL,
                event TEXT NOT NULL,
                correlation_id TEXT,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                mission_state TEXT NOT NULL,
                shot_mode TEXT NOT NULL,
                ready_to_record INTEGER NOT NULL,
                lock_state TEXT NOT NULL,
                recording_state TEXT NOT NULL,
                active_clip_id TEXT,
                battery_percent REAL,
                payload_json TEXT NOT NULL
            );
            """
        )
        self._ensure_schema_version()
        self._conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
            ("session_id", self.session_id),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
            ("schema_version", str(self.SCHEMA_VERSION)),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
            ("created_at", str(int(time.time()))),
        )
        self._conn.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")
        self._conn.commit()

    def _ensure_schema_version(self) -> None:
        user_version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if user_version not in (0, self.SCHEMA_VERSION):
            raise RuntimeError(
                f"Unsupported session log schema version {user_version}; "
                f"expected {self.SCHEMA_VERSION}"
            )

        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?",
            ("schema_version",),
        ).fetchone()
        if row is not None:
            existing = int(row["value"])
            if existing != self.SCHEMA_VERSION:
                raise RuntimeError(
                    f"Unsupported session log schema version {existing}; "
                    f"expected {self.SCHEMA_VERSION}"
                )

    def record_event(self, event: ProductEvent) -> None:
        """Persist a product event to SQLite and the JSONL sidecar."""
        payload_json = json.dumps(event.payload, sort_keys=True)
        event_json = json.dumps(event.to_dict(), sort_keys=True)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO events(
                    seq, timestamp, monotonic_time, source, event, correlation_id, payload_json
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.seq,
                    event.timestamp,
                    event.monotonic_time,
                    event.source,
                    event.event,
                    event.correlation_id,
                    payload_json,
                ),
            )
            self._append_jsonl_line(self.events_path, event_json)

    def record_snapshot(self, snapshot: ProductSnapshot) -> None:
        """Persist a product snapshot to the snapshots table."""
        payload = snapshot.to_dict()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO snapshots(
                    timestamp, mission_state, shot_mode, ready_to_record, lock_state,
                    recording_state, active_clip_id, battery_percent, payload_json
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.timestamp,
                    payload["mission"]["mission_state"],
                    payload["mission"]["shot_mode"],
                    int(payload["mission"]["ready_to_record"]),
                    payload["lock"]["lock_state"],
                    payload["recording_state"],
                    snapshot.active_clip_id,
                    snapshot.battery_percent,
                    json.dumps(payload, sort_keys=True),
                ),
            )

    def _append_jsonl_line(self, path: Path, payload_json: str) -> None:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(payload_json + "\n")

    def iter_events(self) -> Iterator[ProductEvent]:
        """Yield all stored events in sequence order."""
        cursor = self._conn.execute(
            """
            SELECT seq, timestamp, monotonic_time, source, event, correlation_id, payload_json
            FROM events
            ORDER BY seq ASC
            """
        )
        for row in cursor:
            yield ProductEvent(
                seq=row["seq"],
                timestamp=row["timestamp"],
                monotonic_time=row["monotonic_time"],
                session_id=self.session_id,
                source=row["source"],
                event=row["event"],
                payload=json.loads(row["payload_json"]),
                correlation_id=row["correlation_id"],
            )

    def iter_snapshots(self) -> Iterator[dict[str, Any]]:
        """Yield all stored snapshots as deserialized dicts."""
        cursor = self._conn.execute(
            """
            SELECT id, timestamp, payload_json
            FROM snapshots
            ORDER BY id ASC
            """
        )
        for row in cursor:
            payload = json.loads(row["payload_json"])
            payload["id"] = row["id"]
            yield payload

    def timeline(self) -> list[dict[str, Any]]:
        """Return a merged, time-sorted list of events and snapshots."""
        entries: list[dict[str, Any]] = []

        for event in self.iter_events():
            entries.append({"kind": "event", **event.to_dict()})

        for snapshot in self.iter_snapshots():
            entries.append({"kind": "snapshot", **snapshot})

        entries.sort(
            key=lambda item: (
                item["timestamp"],
                0 if item["kind"] == "event" else 1,
                item.get("seq", item.get("id", 0)),
            )
        )
        return entries

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()

    def __enter__(self) -> "SessionLog":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
