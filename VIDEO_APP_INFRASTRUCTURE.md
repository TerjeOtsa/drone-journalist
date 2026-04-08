# Autonomous Follow-and-Film Drone - Video, App, and Infrastructure

## Scope and Product Goal

This module owns everything that makes the drone usable as a filming product:

- Fast launch and fast reconnect.
- Live preview on the phone.
- Trustworthy recording status.
- Reliable clip retrieval even with unstable connectivity.

This module does not own flight control or tracking algorithms. It consumes their outputs and turns them into a usable product experience.

## Assumptions

- Single drone, single operator, single phone.
- Onboard recording is the source of truth.
- The companion computer is Linux-based and has hardware video encode support.
- Autonomy and perception already publish their state on a local interface.
- Link loss must not interrupt onboard recording.

## Current Repo Compatibility

This repo already contains a working autonomy prototype built around Python dataclasses and an in-process event bus:

- Perception input: `interfaces.schemas.TargetTrack`
- Autonomy status: `interfaces.schemas.MissionStatus`
- Operator commands: `interfaces.schemas.AppCommand`
- Safety status: `interfaces.schemas.SafetyStatus`
- Internal events: `interfaces.schemas.SystemEvent` via `interfaces.event_bus.bus`

The design below is the product-grade target architecture for the video/app/infrastructure slice. In the current codebase, this slice should integrate through adapters rather than redefining autonomy semantics:

- `TargetTrack` is adapted into lock and confidence state for the app/video layer.
- `MissionStatus` is adapted into mission/recording readiness state for the app/video layer.
- App API commands are translated into the current `AppCommand` actions.
- Current `SystemEvent` and `SafetyStatus` objects are mirrored into durable product logs.

## Top-Level Architecture

### On-device services

| Service | Language | Responsibility |
|---|---|---|
| `video-service` | Rust + GStreamer | Camera ingest, encode, pre-roll, clip segmentation, local recording, preview stream |
| `app-service` | Rust (`axum`) | HTTPS API, WebSocket telemetry, WebRTC signaling, clip download endpoints, pairing/session handling |
| `session-service` | Rust | Session lifecycle, event log sink, replay export, health snapshots |
| `config-service` | Rust | Parameter validation, profile storage, versioning, config apply/rollback |
| `nats-server` | Off-the-shelf | Local event bus for pub/sub, request/reply, and state fanout |
| `autonomy-*` | Existing | Publishes mission state, ready-to-record, safety events |
| `perception-*` | Existing | Publishes lock state and confidence |

### Mobile app

| Component | Choice | Reason |
|---|---|---|
| App client | Flutter | One product codebase for iOS and Android, fast iteration for a small system |
| Local connectivity | BLE + Wi-Fi | BLE for discovery/bootstrap, Wi-Fi for preview/control/download throughput |
| Preview | WebRTC | Lowest-latency, adaptive bitrate, mobile-native support |
| Telemetry | WebSocket over HTTPS | Simple reconnection and snapshot sync |
| Clip download | HTTPS with resumable range requests | Reliable on unstable links and easy to debug |

### Runtime topology

```text
Camera Sensor
  -> video-service
      -> archival encoder -> pre-roll ring -> segment writer -> clip store
      -> preview encoder -> WebRTC sender -> phone

Autonomy modules ----\
Perception modules ---+-> NATS event bus -> app-service -> phone UI
Safety module -------/                  -> session-service -> logs/replay
                                      -> config-service
                                      -> video-service
```

## 1. Video Pipeline

### Pipeline design

The camera path is split into two products from the same ingest stream:

1. Primary archive path
   - Purpose: source-of-truth recording.
   - Codec: H.265 by default, H.264 fallback if hardware decode compatibility or thermal state requires it.
   - Resolution: default `3840x2160@30`.
   - GOP: 1 second keyframe interval.
   - Bitrate: target `35-60 Mbps` depending on profile.
   - Output: segmented fragmented MP4 files plus a clip manifest.

2. Secondary preview path
   - Purpose: low-latency operator preview.
   - Codec: H.264 for maximum phone compatibility.
   - Resolution: default `1280x720@30`.
   - Bitrate: adaptive `1.0-4.0 Mbps`.
   - Output: WebRTC video track.

### Why this split

- Archive quality is optimized for final footage, not transport.
- Preview quality is optimized for latency and resilience, not permanence.
- A link drop affects preview only. The archive path continues locally.

### GStreamer graph

```text
camera-src
  -> colorspace/isp
  -> tee
     -> queue
        -> hw-h265enc
        -> fragmenter
        -> pre-roll ring buffer
        -> clip segment writer
     -> queue
        -> scale
        -> hw-h264enc low-latency profile
        -> webrtcbin
```

### Segment format

- Authoritative storage unit: `10 s` fragmented MP4 segment.
- Authoritative clip object: manifest + ordered segment list.
- Segment close policy: flush and `fsync` on each segment close.
- Failure model: at worst, the currently open segment is lost on sudden power failure; prior segments remain valid.

### Storage layout

```text
/data/sessions/<session_id>/
  session.sqlite
  events.binpb
  clips/<clip_id>/
    clip_manifest.json
    seg_000000.mp4
    seg_000001.mp4
    seg_000002.mp4
  diagnostics/
    service.log
    health.json
```

## 2. Recording Logic

### Recording state machine

| State | Meaning |
|---|---|
| `IDLE` | Camera not armed for recording |
| `ARMED` | Archive encoder running, pre-roll filling, no active clip |
| `STARTING` | Start condition met, manifest created, pre-roll being flushed |
| `RECORDING_CONFIRMED` | First post-start keyframe written and manifest persisted |
| `STOPPING` | Stop condition met, current segment being finalized |
| `ERROR` | Recorder cannot guarantee integrity |

### Pre-roll

- Default: `8 seconds`.
- Implementation: keep the last few archive fragments in RAM, already encoded at archive quality.
- Benefit: the clip starts before the formal trigger without waiting for a new encoder startup.
- Configurable per profile.

### Auto-start policy

Auto-record starts when all of the following are true:

- Autonomy publishes `MissionStatus.state = film`.
- Autonomy publishes `ready_to_record = true` for at least `300 ms`.
- Recorder health is `OK`.
- Storage has enough free space for at least one new clip window.

Perception confidence is attached to the clip and shown in the UI, but the video layer does not second-guess autonomy by adding an independent start gate on top of `ready_to_record`.

Manual start from the app always overrides `ready_to_record`.

### Auto-stop policy

Auto-record stops when any of the following remain true for the hold time:

- `ready_to_record = false` for `2000 ms`.
- `MissionStatus.state != film`.
- User explicitly stops recording.

Safety events use this policy:

- `REDUCE_SPEED` or `INCREASE_DIST`: keep recording and tag the clip.
- `RETURN_HOME`: keep recording unless storage or thermal state becomes critical.
- `LAND_NOW` or `EMERGENCY_STOP`: stop accepting new clips and finalize the active clip as safely as possible.

### Trustworthy recording UX

The app must not show a solid recording indicator just because a command was accepted.

The app shows:

- `arming`: hollow red dot, recorder warming or flushing pre-roll.
- `recording_confirmed`: solid red dot plus elapsed timer, only after the recorder emits `RECORDING_CONFIRMED`.
- `degraded`: red dot plus warning badge if frames are being dropped or storage is close to full.

### Clip metadata

Each clip manifest stores:

- `clip_id`
- `session_id`
- `profile`
- `started_at_utc`
- `ended_at_utc`
- `start_reason`
- `stop_reason`
- `pre_roll_sec`
- `shot_mode`
- `lock_state_at_start`
- `identity_confidence_min`
- `tracking_confidence_min`
- `effective_confidence_min`
- `effective_confidence_avg`
- `effective_confidence_max`
- `safety_events[]`
- `segment_count`
- `total_bytes`
- `sha256_manifest`
- `log_offset_start`
- `log_offset_end`

Example:

```json
{
  "clip_id": "clip_2026-04-07T13-12-44Z_0001",
  "session_id": "sess_2026-04-07T13-11-58Z",
  "profile": "walk",
  "started_at_utc": "2026-04-07T13:12:44.210Z",
  "ended_at_utc": "2026-04-07T13:13:16.901Z",
  "start_reason": "ready_to_record",
  "stop_reason": "ready_timeout",
  "pre_roll_sec": 8,
  "shot_mode": "walk_and_talk",
  "lock_state_at_start": "locked",
  "identity_confidence_min": 0.66,
  "tracking_confidence_min": 0.62,
  "effective_confidence_min": 0.62,
  "effective_confidence_avg": 0.88,
  "effective_confidence_max": 0.97,
  "safety_events": ["wind_warn"],
  "segment_count": 4,
  "total_bytes": 212334188,
  "sha256_manifest": "8f6d...",
  "log_offset_start": 118224,
  "log_offset_end": 151009
}
```

## 3. Drone <-> Phone Communication System

### Connectivity model

- BLE is used for first discovery, secure bootstrap, and emergency "find my drone" interactions.
- Wi-Fi is used for preview, control, telemetry, and downloads.
- The drone acts as the default Wi-Fi AP in field mode to remove dependency on public infrastructure.
- mDNS advertises `drone.local` on the local network.

### Pairing and authentication

#### First-time pairing

1. Phone discovers the drone over BLE.
2. User scans a QR code on the drone body containing `drone_id` and the factory public-key fingerprint.
3. User presses a hardware pair button on the drone for physical possession proof.
4. Phone and drone perform an ECDH exchange over BLE.
5. Drone sends Wi-Fi credentials and issues a paired-device certificate bound to that phone.

#### Session authentication

- After pairing, the phone connects to drone Wi-Fi and opens TLS to `https://drone.local`.
- The paired device certificate is used for mutual authentication.
- `app-service` issues a short-lived session token for REST, WebSocket, and WebRTC signaling.

This gives:

- Fast reconnect without re-pairing.
- Local-only trust without a cloud dependency.
- Clean revocation by deleting the paired device certificate on the drone.

### Control and telemetry transport split

| Function | Transport | Reason |
|---|---|---|
| Commands | HTTPS `POST` | Simple, auditable, idempotent |
| Telemetry and app state | WebSocket | Continuous updates and fast resync |
| Preview video | WebRTC | Low latency and adaptive bitrate |
| Clip retrieval | HTTPS `GET` + range requests | Resumable and reliable |

### Reconnection behavior

If the phone disconnects:

- Archive recording continues locally without change.
- WebRTC preview retries with exponential backoff.
- WebSocket reconnects and immediately requests a fresh state snapshot.
- The app asks for clip deltas using the highest seen clip sequence.
- Partial clip downloads resume with HTTP range requests and checksum validation.
- Under the current autonomy defaults in `config/parameters.py`, link-loss safety may drive `RETURN_HOME` after `3.0 s` and `LAND_NOW` after `8.0 s`; the product layer must preserve recording and resync cleanly across those transitions.

If the phone reconnects after a longer drop:

- The first response from `app-service` is a full snapshot, not a diff.
- The app does not assume missed commands succeeded unless it received a command ack.
- Recorder state and clip list always come from the drone, never from optimistic local app state.

## 4. Mobile App API

### Command API

All commands are idempotent and require a `command_id`.

| Endpoint | Purpose |
|---|---|
| `POST /v1/commands/launch` | Request takeoff / mission start |
| `POST /v1/commands/relock` | Request a new target lock or recovery behavior |
| `POST /v1/commands/mode` | Change shot mode |
| `POST /v1/commands/record/start` | Manual record override |
| `POST /v1/commands/record/stop` | Manual stop |
| `POST /v1/commands/config/profile` | Apply `stand_up`, `walk`, or `safe` |
| `POST /v1/commands/ping` | Link health / latency check |

Example:

```http
POST /v1/commands/launch
Content-Type: application/json
Authorization: Bearer <session-token>
```

```json
{
  "command_id": "cmd_9f40d2da",
  "requested_at_utc": "2026-04-07T13:12:40.101Z",
  "shot_mode": "walk_and_talk"
}
```

Response:

```json
{
  "command_id": "cmd_9f40d2da",
  "accepted": true,
  "correlation_id": "corr_0db5f7a2"
}
```

### Snapshot API

`GET /v1/state`

Returns the full current product state used on app cold-start or reconnect:

```json
{
  "drone_id": "drn_001",
  "session_id": "sess_2026-04-07T13-11-58Z",
  "mission_state": "film",
  "shot_mode": "walk_and_talk",
  "ready_to_record": true,
  "recording_state": "recording_confirmed",
  "active_clip_id": "clip_2026-04-07T13-12-44Z_0001",
  "lock_state": "locked",
  "lock_confidence": {
    "identity": 0.93,
    "tracking": 0.91,
    "effective": 0.91
  },
  "battery_percent": 63,
  "storage_minutes_remaining": 47,
  "link_rssi_dbm": -59,
  "preview_latency_ms": 132,
  "safety_level": "ok",
  "profile": "walk"
}
```

### Live telemetry stream

`GET /v1/ws`

WebSocket events are JSON envelopes with:

- `type`
- `seq`
- `session_id`
- `source`
- `mono_time_ns`
- `payload`

`app-service` also pushes a full snapshot immediately after the socket opens.

### Preview setup

Preview uses REST for signaling and WebRTC for media:

1. App calls `POST /v1/preview/session`.
2. Drone returns SDP offer and ICE candidates.
3. App posts SDP answer to `POST /v1/preview/session/{id}/answer`.
4. Preview starts automatically.

### Clip retrieval API

| Endpoint | Purpose |
|---|---|
| `GET /v1/clips?after_seq=<n>` | Incremental manifest sync |
| `GET /v1/clips/{clip_id}` | Clip manifest |
| `GET /v1/clips/{clip_id}/segments/{index}` | Segment download with range support |
| `GET /v1/clips/{clip_id}/export.mp4` | Optional stitched export |

Retrieval policy:

- In flight: sync manifests only by default.
- On strong link or after landing: background download full segments.
- App validates segment checksums against the manifest before marking the clip complete.

### Telemetry shown in the mobile UI

- Mission state
- Ready-to-record state
- Recording state and duration
- Lock state and confidence
- Safety banner and latest safety event
- Battery percentage and storage remaining
- Link quality and preview latency
- Active profile

### Lock-status UI signals

| Lock state | UI treatment |
|---|---|
| `locked`, effective confidence `>= 0.80` | Green subject outline, stable badge |
| `candidate`, effective confidence `< 0.80` | Amber pulse and "confirming subject" label |
| `weak`, effective confidence `0.35-0.79` | Amber outline and "tracking weak" label |
| `lost` | Red banner and optional auto-relock affordance |

## 5. Event Bus

### Bus choice

Use a staged bus design:

- Current repo: keep using the existing in-process `interfaces.event_bus.EventBus` for the autonomy prototype and simulation.
- Product target: use local NATS for cross-process video, app, logging, and configuration services on the companion computer.

Reasons for NATS in the product target:

- Small operational footprint.
- Supports pub/sub and request/reply.
- Works well with existing Python modules and Rust services.
- Easy to observe and easy to replay from logs.

Compatibility rule:

- The video/app layer owns the adapter between the current Python objects and the production bus subjects below.
- Semantic fields must stay aligned with `TargetTrack`, `MissionStatus`, `SafetyStatus`, `AppCommand`, and `SystemEvent`.

### Topic model

Separate fast-changing state from durable events:

- `state.*`: latest-wins, no persistence requirement on the bus itself.
- `event.*`: important transitions and incidents, always written to the session log.
- `cmd.*`: operator commands and service-to-service requests, idempotent.

### Envelope

Internal bus messages use protobuf inside a common envelope:

```proto
message Envelope {
  string type = 1;
  uint32 schema_version = 2;
  string session_id = 3;
  string source = 4;
  uint64 seq = 5;
  uint64 mono_time_ns = 6;
  int64 wall_time_unix_ms = 7;
  string correlation_id = 8;
  bytes payload = 9;
}
```

### Required message types

| Subject | Produced by | Key fields |
|---|---|---|
| `state.mission.v1` | Autonomy adapter | `mission_state`, `shot_mode`, `ready_to_record`, `stability`, `safety_override`, `message` |
| `state.lock.v1` | Perception adapter | `lock_state`, `identity_confidence`, `tracking_confidence`, `effective_confidence`, `target_id` |
| `event.safety.v1` | Safety adapter | `override`, `reasons` |
| `state.recording.v1` | Video service | `recording_state`, `active_clip_id`, `dropped_frames`, `free_space_mb` |
| `event.clip.started.v1` | Video service | `clip_id`, `start_reason`, `pre_roll_sec` |
| `event.clip.segment.v1` | Video service | `clip_id`, `segment_index`, `path`, `bytes`, `sha256` |
| `event.clip.finalized.v1` | Video service | `clip_id`, `duration_ms`, `segment_count`, `total_bytes` |
| `state.link.v1` | App service | `connected`, `rssi_dbm`, `rtt_ms`, `preview_bitrate_kbps` |
| `cmd.autonomy.launch.v1` | App service | `command_id`, `requested_shot_mode`, `actor` |
| `cmd.autonomy.mode.v1` | App service | `command_id`, `shot_mode`, `actor` |
| `cmd.autonomy.relock.v1` | App service | `command_id`, `strategy` |
| `cmd.video.record.v1` | App service | `command_id`, `action` |
| `event.config.applied.v1` | Config service | `profile`, `config_version`, `overrides_hash` |

### Synchronization strategy

- In the current prototype, `EventBus` delivery is synchronous and in-order inside one process.
- Every producer owns a monotonic per-source sequence counter.
- Every message carries device monotonic time plus wall-clock UTC.
- State consumers use `seq` to reject out-of-order updates.
- Commands are idempotent by `command_id`.
- `app-service` keeps an in-memory state cache so reconnecting clients get a snapshot immediately.
- `session-service` records all events in arrival order plus original source timestamp for replay fidelity.
- Sequence counters are assigned at the adapter boundary when current in-process messages are mirrored onto the product bus.

## 6. Logging and Replay

### What gets logged

- All `event.*` messages.
- Periodic snapshots of `state.*` messages.
- API requests, command acks, and command failures.
- WebRTC stats every few seconds while preview is active.
- Recorder health: dropped frames, encoder resets, disk write latency, storage headroom.
- Link metrics: RSSI, RTT, bitrate, reconnect count.
- Config changes and active profile.

### Log format

Use a per-session bundle:

- `session.sqlite` for indexed metadata, clips, commands, and health tables.
- `events.binpb` as an append-only protobuf log of raw bus envelopes.
- Clip manifests and media segments in the same session folder.

This gives:

- Fast indexed lookups.
- Exact raw-message replay.
- A single portable artifact per flight session.

### Replay flow

`replay-tool <session_dir>`:

1. Reads `events.binpb` and `session.sqlite`.
2. Re-publishes recorded envelopes onto a sandbox NATS bus.
3. Replays timing at `1x`, `10x`, or step-through mode.
4. Optionally serves clip manifests and media for UI playback.

### Debugging workflow

1. Open a session by `session_id`.
2. Inspect the event timeline for `lock`, `ready_to_record`, `recording`, and `safety`.
3. Jump from any clip to its matching log window using `log_offset_start/end`.
4. Compare preview-link degradation with recorder health to verify the archive was unaffected.
5. Export a session bundle for offline engineering analysis.

## 7. Configuration System

### Storage model

- Factory defaults are bundled in the image.
- Field-tunable parameters live in a signed versioned config file.
- Runtime overrides are stored in `session.sqlite` and applied through `config-service`.
- In the current repo, these product profiles overlay the existing `SystemConfig` values defined in `config/parameters.py`.

### Config schema

```yaml
schema_version: 3
config_version: "2026.04.07-01"
min_app_version: "1.4.0"
min_autonomy_schema: 2
profiles:
  stand_up:
    pre_roll_sec: 8
    preview_max_bitrate_kbps: 1500
    record_ready_hold_ms: 300
    record_stop_hold_ms: 3000
  walk:
    pre_roll_sec: 8
    preview_max_bitrate_kbps: 2500
    record_ready_hold_ms: 300
    record_stop_hold_ms: 2000
  safe:
    pre_roll_sec: 4
    preview_max_bitrate_kbps: 1200
    record_ready_hold_ms: 500
    record_stop_hold_ms: 3500
```

### Profiles

| Profile | Intended use |
|---|---|
| `stand_up` | Mostly static subject, conservative record transitions, lower preview bitrate |
| `walk` | Moving subject, balanced responsiveness and clip continuity |
| `safe` | Maximum conservatism, longer record holds, lower bandwidth, and lower thermal load |

These profiles are not the same thing as autonomy shot modes:

- Profiles tune recording, preview, and network behavior.
- Shot mode remains the autonomy concept already present in the repo: `standup`, `walk_and_talk`, `wide_safety`, or `orbit`.

### Versioning and apply rules

- Every config has `schema_version` and `config_version`.
- `config-service` validates compatibility before apply.
- A failed apply never partially updates the system.
- Every successful apply emits `event.config.applied.v1`.
- The active config fingerprint is included in each clip manifest.

## 8. Interface Contracts

### Current code mapping

| Current repo object | Product-layer meaning |
|---|---|
| `TargetTrack.lock_state` | `state.lock.v1.lock_state` |
| `TargetTrack.identity_confidence` | `state.lock.v1.identity_confidence` |
| `TargetTrack.tracking_confidence` | `state.lock.v1.tracking_confidence` |
| `min(identity_confidence, tracking_confidence)` | `state.lock.v1.effective_confidence` and default UI confidence |
| `MissionStatus.state` | `state.mission.v1.mission_state` |
| `MissionStatus.shot_mode` | `state.mission.v1.shot_mode` |
| `MissionStatus.ready_to_record` | recorder auto-start input |
| `SafetyStatus.active_override` or `SystemEvent(source=\"safety\", event=\"safety_override\")` | `event.safety.v1` |
| `AppCommand(action=\"start\")` | generated from `POST /v1/commands/launch` |
| `AppCommand(action=\"set_shot_mode\")` | generated from `POST /v1/commands/mode` |
| `AppCommand(action=\"relock\")` | generated from `POST /v1/commands/relock` |

### Inputs from autonomy

`state.mission.v1`

```json
{
  "mission_state": "film",
  "shot_mode": "walk_and_talk",
  "ready_to_record": true,
  "stability": "nominal",
  "safety_override": "none",
  "message": "film"
}
```

`event.safety.v1`

```json
{
  "override": "return_home",
  "reasons": ["battery low: 18%"]
}
```

### Inputs from perception

`state.lock.v1`

```json
{
  "lock_state": "locked",
  "identity_confidence": 0.93,
  "tracking_confidence": 0.91,
  "effective_confidence": 0.91,
  "target_id": "subject_primary"
}
```

### Outputs to the phone

`state.recording.v1`

```json
{
  "recording_state": "recording_confirmed",
  "active_clip_id": "clip_2026-04-07T13-12-44Z_0001",
  "free_space_mb": 21568,
  "dropped_frames": 0
}
```

`event.clip.finalized.v1`

```json
{
  "clip_id": "clip_2026-04-07T13-12-44Z_0001",
  "duration_ms": 32691,
  "segment_count": 4,
  "total_bytes": 212334188
}
```

## 9. Message Flows

### Launch and auto-record

```text
Phone
  -> POST /v1/commands/launch
app-service
  -> cmd.autonomy.launch.v1
autonomy
  -> state.mission.v1 (takeoff)
video-service
  -> state.recording.v1 (armed)
autonomy
  -> state.mission.v1 (film, ready_to_record=true)
video-service
  -> event.clip.started.v1
  -> state.recording.v1 (recording_confirmed)
app-service
  -> WebSocket state update
phone UI
  -> solid REC indicator + timer
```

### Link drop and reconnect

```text
Wi-Fi degrades
  -> WebRTC preview stalls
  -> WebSocket disconnects
video-service
  -> keeps recording locally
session-service
  -> keeps logging
phone reconnects
  -> GET /v1/state
  -> GET /v1/clips?after_seq=<last_seen>
  -> POST /v1/preview/session
app-service
  -> returns full snapshot and clip delta
phone UI
  -> state catches up from drone truth
```

### Clip sync after flight

```text
phone
  -> GET /v1/clips?after_seq=0
app-service
  -> returns manifests
phone
  -> downloads missing segments with range requests
  -> validates sha256 per segment
  -> optionally requests /export.mp4
drone
  -> keeps manifest + segments as source of truth
```

## 10. Failure Handling

| Failure | Behavior |
|---|---|
| Preview encoder crash | Restart preview path only; archive path stays up |
| App disconnect | Keep recording and keep logging |
| WebRTC negotiation failure | Fall back to still-healthy telemetry and clip sync |
| Low storage | Warn early, then prevent new auto-clips before active clip integrity is at risk |
| Segment write failure | Move recorder to `ERROR`, publish safety-grade alert, preserve prior segments |
| Clock jump | Use monotonic time for ordering and replay |

## 11. Recommended Implementation Choices

### Protocols

- Internal bus: NATS with protobuf payloads.
- Phone API: HTTPS/JSON.
- Telemetry: WebSocket JSON envelopes.
- Preview: WebRTC.
- Clip download: HTTPS range requests.

### Languages

- Companion services: Rust.
- Video/media graph: GStreamer controlled from Rust.
- Existing autonomy/perception: keep current language, adapt to NATS protobuf contracts.
- Mobile app: Flutter.

### Storage formats

- Event log: binary protobuf stream.
- Session index: SQLite.
- Media: segmented fragmented MP4.
- Clip metadata: JSON manifest plus indexed rows in SQLite.

## 12. Why This Meets the Product Goal

- Launch quickly: cached pairing, auto Wi-Fi reconnect, snapshot-first app startup.
- See live preview: WebRTC preview path tuned separately from archive.
- Trust recording: recorder-confirmed UI state instead of optimistic app state.
- Retrieve clips reliably: onboard manifest plus resumable segment download and checksums.
- Survive unstable links: archive and logs are local-first, preview and app state resync cleanly.
