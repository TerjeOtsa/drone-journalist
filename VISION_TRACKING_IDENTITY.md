# Vision / Tracking / Identity Architecture

## Purpose

This document defines the perception contract for a compact follow-and-film drone that must lock onto one journalist, keep that identity over time, and fail safely when certainty drops.

Design priorities:

1. Consistency over aggressiveness.
2. Single subject only.
3. Never switch to a different person on weak evidence.
4. Face is optional and only used for lock/relock confirmation.
5. When uncertain, degrade to `weak` or `lost` instead of guessing.

## Module Split

### Perception Module

Owns:

- frame conditioning
- person detection
- multi-person track generation
- pose and body representation
- world-position estimation when telemetry supports it

### Identity & Lock Module

Owns:

- enrollment handling
- initial journalist lock
- identity maintenance over time
- relock after occlusion or temporary loss
- confidence fusion and lock-state transitions

## 1. Perception Pipeline

### 1.1 Frame Conditioning

Run lightweight pre-processing before inference:

- gyro/gimbal-assisted image stabilization when available
- exposure normalization and local contrast boost for harsh lighting
- blur score estimation
- visibility score estimation for face and full body

These scores do not directly identify the target; they only modulate confidence and model scheduling.

### 1.2 Person Detection

Use a lightweight person detector optimized for edge deployment.

Recommended behavior:

- detector input: `640x384` or `640x480`
- detector cadence: `10-15 Hz`
- confidence calibration per environment, not raw logits
- detector outputs all visible people, not just the target

Rationale: even in a single-subject system, we must model nearby distractors to avoid switching to the wrong person.

### 1.3 Track Manager

Maintain short-term tracks for every detected person.

Each track stores:

- track ID
- bounding box history
- per-frame detector confidence
- motion state from a constant-velocity Kalman filter
- appearance embedding history
- pose/keypoint history
- occlusion and visibility flags

Association between detections and existing tracks uses:

- motion gating from the Kalman prediction
- IoU overlap
- appearance similarity
- pose consistency

The target tracker must continue through short detector dropouts using prediction, but prediction-only mode is bounded in time and confidence.

### 1.4 Pose / Body Representation

Run a lightweight pose or keypoint head on the top few tracks.

Use pose for:

- torso and foot-point estimation
- visibility scoring
- body orientation
- coarse gait or shape consistency
- better world-position estimation when feet are visible

Use a body appearance encoder for re-identification:

- embedding from torso/full-body crop
- updated only from high-confidence frames
- slow online adaptation to lighting and pose
- no update during occlusion, blur, or ambiguity

This online update rule is important; otherwise the model drifts onto the wrong person.

### 1.5 Optional Face Verification

Face is event-triggered, not continuous.

Use it only when:

- initial lock at launch
- relock after `weak` or `lost`
- user-initiated re-enrollment

Face verification requires a quality gate:

- frontal or near-frontal face
- sufficient face size
- acceptable blur and exposure

If face quality is poor, the system must ignore face evidence instead of forcing a decision.

### 1.6 World Position Estimation

Output `target_position_world` only when telemetry and camera geometry make it credible.

Preferred method:

- estimate image foot-point from pose or box bottom center
- project ray through camera intrinsics
- intersect with ground plane using drone pose, altitude, and camera extrinsics

When feet are not visible:

- fall back to monocular range from person height prior only if confidence remains low-marked
- otherwise return `null` for world position

Rule: bad world position is worse than missing world position.

### 1.7 Integration Constraints With Current Project

The current autonomy stack in this repo already consumes a shared
`TargetTrack` object from `interfaces/schemas.py`.

Important implications from the existing code:

- the mission state machine keys primarily off `lock_state`
- the shot controller only follows when `target_position_world` is present
- if `target_position_world` is missing, the shot controller will hover rather than infer depth from image coordinates

So for this project, image-only tracking is useful for lock maintenance and diagnostics,
but sustained active follow requires a credible world estimate at the publish rate used by autonomy.

## 2. Identity System

### 2.1 Enrollment Inputs

Optional app enrollment should provide:

- one face image
- one full-body image
- optional 1-2 second body turn clip
- optional phone beacon ID

These create:

- face template
- body embedding prototype
- expected height range
- optional phone identity token

### 2.2 Initial Lock at Launch

At launch, score each visible person as a target candidate.

Candidate score uses:

- face verification against enrollment, if visible
- body embedding similarity to enrollment
- temporal proximity to launch moment
- spatial prior near the expected launch handoff location
- phone proximity likelihood, if available

Initial lock rule:

- require the best candidate to exceed the lock threshold
- require a margin over the second-best candidate
- require stability for consecutive frames

Recommended thresholds:

- best candidate fused score `>= 0.78`
- score margin to runner-up `>= 0.15`
- stable for `12` consecutive frames at `30 FPS` or equivalent

If these conditions are not met within a short acquisition window, remain in `candidate` and request help from the app rather than guessing.

### 2.3 Identity Maintenance

Once locked, identity is maintained primarily by:

- track continuity
- body appearance similarity
- pose and shape consistency
- motion continuity over time
- optional phone proximity

Face verification is a bonus confirmation cue, not the main loop.

The system keeps a protected target profile:

- locked target track ID
- long-term enrollment embedding
- short-term appearance prototype
- last strong pose profile
- last known world state
- last positive face verification time
- last positive phone support time

### 2.4 Fusion Sources

Identity evidence comes from:

- `E_body`: body appearance similarity to protected target profile
- `E_face`: face verification score when a valid face crop exists
- `E_phone`: phone proximity likelihood
- `E_temp`: temporal continuity from last locked state
- `E_pose`: pose and body-shape consistency

Tracking evidence comes from:

- `T_det`: calibrated detector confidence
- `T_assoc`: association confidence against the existing target track
- `T_motion`: consistency with predicted motion
- `T_vis`: visible-body fraction
- `T_quality`: image quality after blur/exposure penalties

### 2.5 Lock States

The target state machine has four states:

- `candidate`: one or more plausible journalist candidates exist, but identity is not yet trusted
- `locked`: strong identity and track continuity; safe to follow
- `weak`: target is probably still the same person, but evidence has degraded
- `lost`: no safe target hypothesis remains

### 2.6 State Transition Rules

Recommended operating rules:

- `candidate -> locked` when `identity_confidence >= 0.75`, `tracking_confidence >= 0.70`, ambiguity margin is acceptable, and the same candidate is stable for `0.4-0.6 s`
- `locked -> weak` when either confidence drops below strong-lock levels for `> 5` frames, or short occlusion begins
- `weak -> locked` when both confidences recover above relock thresholds for `>= 10` consecutive frames
- `weak -> lost` when no target association exists for `> 1.5 s`, or ambiguity persists without recovery
- `lost -> candidate` when a plausible match reappears but is not yet trusted
- `lost -> locked` is allowed only after the full lock criteria are satisfied again

There is no direct `locked -> another person`.

Instead:

1. degrade current target to `weak` or `lost`
2. evaluate relock candidates
3. require fresh lock evidence

This no-switch rule is the main safety mechanism against identity jumps in crowds.

## 3. Confidence Model

All confidences are in `[0.0, 1.0]`.

### 3.1 Identity Confidence

Compute a fused identity score from active cues:

```text
S_id_base =
  0.35 * E_body +
  0.20 * E_temp +
  0.10 * E_pose +
  0.15 * E_phone +
  0.20 * E_face
```

If a cue is unavailable, renormalize over active weights.

Apply two modifiers:

- `ambiguity_penalty`: if the top two candidates are too close
- `time_decay_id`: since last positive identity evidence

```text
identity_confidence = S_id_base * ambiguity_penalty * time_decay_id
```

Recommended rules:

- if runner-up is within `0.10` of the best candidate, cap `identity_confidence` at `0.55`
- `time_decay_id = exp(-dt / 2.0 s)` since the last positive identity cue

Identity decays slowly because body continuity can survive brief occlusion.

### 3.2 Tracking Confidence

```text
S_track_base =
  0.30 * T_det +
  0.30 * T_assoc +
  0.20 * T_motion +
  0.10 * T_vis +
  0.10 * T_quality
```

Apply prediction decay when the tracker is carrying state without a fresh detection:

```text
tracking_confidence = S_track_base * exp(-dt / 0.6 s)
```

Tracking decays faster than identity because localization becomes unsafe quickly.

### 3.3 Thresholds

Recommended thresholds:

- strong lock: `identity_confidence >= 0.75` and `tracking_confidence >= 0.70`
- weak lock: `identity_confidence in [0.35, 0.75)` or `tracking_confidence in [0.25, 0.70)`
- lost: `tracking_confidence < 0.25`, or no valid association for `> 1.5 s`, or ambiguity prevents safe selection

### 3.4 Online Adaptation Guardrails

Update the short-term appearance prototype only when:

- state is `locked`
- `identity_confidence >= 0.85`
- `tracking_confidence >= 0.80`
- blur is low
- occlusion is low

Do not adapt the identity profile in `weak` or `lost`.

## 4. Failure Handling

### 4.1 Wrong-Person Avoidance

To avoid switching:

- always keep distractor tracks active
- compare the current target to the second-best candidate every cycle
- cap confidence under ambiguity
- forbid direct target handoff between people
- require re-lock evidence after `lost`

### 4.2 Crowded Scenes

In crowds, prioritize conservative behavior:

- keep following only while the current track remains continuous
- if continuity breaks and multiple candidates are similar, enter `weak`
- if ambiguity persists, enter `lost`

Crowded-scene rule:

- never relock on clothing similarity alone after a full loss in a dense crowd

Relock in crowds requires at least one strong cue beyond body appearance:

- face verification
- strong phone proximity
- highly consistent temporal reappearance near the predicted location with clear score margin

### 4.3 Relock Strategy

Relock should expand outward in stages:

1. search near the predicted image and world position
2. widen search to all visible tracks
3. apply face verification if a usable face appears
4. apply phone support if available
5. if still unresolved, request app help

Recommended help triggers:

- no lock acquired within `5 s` of launch
- `weak` persists for `> 2 s`
- `lost` persists for `> 3 s`
- repeated ambiguity in a crowd

### 4.4 App Assistance Requests

The app should be asked for help only with explicit reasons:

- `need_face_view`
- `need_full_body_view`
- `phone_signal_missing`
- `identity_ambiguous`
- `re_enroll_required`

The app may guide the journalist to:

- face the drone
- pause briefly
- step out from behind occlusion
- raise the enrolled phone

## 5. Environmental Robustness

### 5.1 Backlighting

Mitigations:

- exposure normalization before detection
- use body shape, pose, and motion continuity when face is washed out
- maintain lock from temporal continuity if the current track is stable

Limit:

- do not reacquire a lost target from silhouette-only evidence in a crowd

### 5.2 Low Light

Mitigations:

- low-light fine-tuning for detector and re-ID models
- denoise only if latency budget allows
- lower detector cadence before lowering safety thresholds

Limits:

- below roughly `5-10 lux`, face verification is often unusable
- below severe noise and blur conditions, the system may stay `weak` or `lost`

### 5.3 Occlusion

Mitigations:

- short-term motion prediction
- identity decay slower than tracking decay
- relock around the last seen position first

Limits:

- full occlusion beyond `1.5-3.0 s` in a dense crowd cannot be safely solved from monocular video alone
- without phone support or face confirmation, the system should prefer `lost`

### 5.4 Motion Blur

Mitigations:

- blur-aware quality scoring
- shorter shutter preference if camera controls permit
- rely more on temporal continuity during brief blur spikes

Limit:

- sustained blur should prevent both identity adaptation and fresh relock

## 6. Output Interface

Required output contract, once per perception update:

```json
{
  "timestamp_ms": 1712443200123,
  "frame_id": 4821,
  "lock_state": "candidate",
  "lost_target": false,
  "identity_confidence": 0.68,
  "tracking_confidence": 0.81,
  "target_position_image": {
    "valid": true,
    "cx_norm": 0.52,
    "cy_norm": 0.61,
    "w_norm": 0.18,
    "h_norm": 0.42,
    "footpoint_norm": [0.51, 0.82],
    "velocity_px_s": [24.0, -6.0],
    "bbox_confidence": 0.88
  },
  "target_position_world": {
    "valid": true,
    "frame": "local_ned",
    "position_m": [3.2, -1.1, 0.0],
    "velocity_mps": [0.8, 0.2, 0.0],
    "covariance_diag": [0.8, 0.8, 1.5],
    "source": "ground_plane_intersection"
  },
  "diagnostics": {
    "target_track_id": "trk_17",
    "best_candidate_score": 0.76,
    "second_best_score": 0.59,
    "occlusion_ratio": 0.18,
    "blur_score": 0.12,
    "face_used": false,
    "phone_used": true,
    "failure_reason": null,
    "assist_request": null
  }
}
```

Field definitions:

- `lock_state`: one of `candidate`, `locked`, `weak`, `lost`
- `lost_target`: `true` only when `lock_state == "lost"`
- `target_position_image.valid`: `false` if no safe image localization exists
- `target_position_world.valid`: `false` if telemetry or geometry is insufficient
- `identity_confidence`: confidence that the tracked person is still the enrolled journalist
- `tracking_confidence`: confidence that the current image/world localization is correct

Schema notes:

- if `lock_state == "lost"`, both position objects may be `valid: false`
- `target_position_world` may be `null` or `valid: false`; downstream logic must accept both
- `diagnostics` is optional for flight control but required for debugging and evaluation

### 6.1 Canonical Autonomy Interface in This Repo

The richer schema above is the internal perception output. The current project
already expects the perception module to publish `TargetTrack` from
`interfaces/schemas.py` at the autonomy boundary.

Canonical mapping:

```python
TargetTrack(
    timestamp=timestamp_ms / 1000.0,
    target_position_image=(cx_norm, cy_norm),
    target_position_world=Vec3(*position_m) if target_position_world["valid"] else None,
    target_velocity_world=Vec3(*velocity_mps) if target_position_world["valid"] else None,
    identity_confidence=identity_confidence,
    tracking_confidence=tracking_confidence,
    lock_state=LockState(lock_state),
    bounding_box=(
        cx_norm - 0.5 * w_norm,
        cy_norm - 0.5 * h_norm,
        w_norm,
        h_norm,
    ) if target_position_image["valid"] else None,
)
```

Compatibility rules:

- `lost_target` is derived, not separately carried, because `lock_state == "lost"` already encodes it
- `target_position_world` should be `None` when not credible; downstream follow logic already treats this conservatively
- `target_position_image` may still be populated when world position is unavailable, but autonomy should not treat it as sufficient for active follow
- `bounding_box` should be populated whenever image localization is valid because it is the safest compact carrier for image geometry in the existing schema

### 6.2 Reduced Lock Event for App / Infrastructure

The video and app infrastructure document currently expects a reduced lock event
with only `lock_state`, `confidence`, and `target_id`.

Recommended reduction from the richer perception state:

```json
{
  "lock_state": "locked",
  "confidence": 0.68,
  "target_id": "subject_primary"
}
```

Reduction rules:

- `confidence = min(identity_confidence, tracking_confidence)` so UI and recording logic stay conservative
- `target_id` should be a stable non-PII token for the enrolled subject, not a face identifier
- app-level translations such as `candidate -> searching` should happen outside the perception core

## 7. Performance Constraints

Assume a compact edge computer in the `10-25 TOPS` class.

Recommended targets:

- camera input: `30 FPS`
- `TargetTrack` publish rate to autonomy: `25 Hz` nominal to match the current project
- internal tracker/state update rate: `20-30 Hz` minimum, ideally every frame
- end-to-end latency: `<= 100 ms` nominal, `<= 150 ms` p95
- detector cadence: `10-15 Hz`
- tracker update cadence: every frame
- pose/re-ID cadence: `8-12 Hz` on the top few tracks
- face verification cadence: on demand, capped at `5 Hz`

Model size expectations:

- person detector: `5-10M` parameters, INT8-friendly
- body re-ID encoder: `3-6M` parameters
- pose/keypoint head: `2-5M` parameters
- face verifier: `1-3M` parameters

Scheduling rule:

- drop optional heads before relaxing safety thresholds

This means:

1. keep the tracker running every frame
2. slow face and pose first
3. reduce detector cadence second
4. never fake confidence to hold frame rate

## 8. Training / Data Strategy

### 8.1 Required Labels

Training data should include:

- person boxes
- multi-frame track IDs
- keypoints or coarse pose
- occlusion and truncation flags
- face visibility and face quality labels
- same-person re-ID labels across time and viewpoint
- world-position ground truth when available

### 8.2 Data Sources

Collect real drone-perspective footage for:

- urban streets
- protests and crowds
- interviews and standups
- indoor entrances and exits
- vehicle approaches
- dawn, dusk, night, noon, and mixed lighting

### 8.3 Journalism-Specific Edge Cases

Prioritize these cases:

- journalist beside interview subject
- journalist walking backward while speaking to camera
- carrying tripod, backpack, mic, or camera bag
- face partially hidden by microphone or headset
- multiple people in similar press clothing
- umbrellas, signs, podiums, police tape, vehicles
- flashing emergency lights
- reflective vests and rain gear
- entering and exiting buildings
- temporary disappear/reappear behind crowds

### 8.4 Augmentation

Use aggressive but realistic augmentation:

- motion blur
- defocus blur
- backlight and lens flare
- underexposure and sensor noise
- rain, mist, dust, snow
- compression artifacts
- partial occlusion

### 8.5 Hard Negative Mining

Build a hard-negative set specifically for false relocks:

- similar clothes
- similar body size
- same scene, different people
- subject after long occlusion

This is critical because wrong-person lock is the worst perception failure.

## 9. Evaluation Metrics

### 9.1 Tracking Accuracy

Measure:

- image-space center error
- target box IoU
- world-position RMSE when ground truth exists
- track continuity while the subject is visible

### 9.2 Identity Stability

Measure:

- ID switch count per hour
- wrong-person lock rate
- fraction of ambiguous cases that correctly degrade to `weak` or `lost`
- false relock rate after occlusion

### 9.3 Recovery

Measure:

- median recovery time after `0.5 s`, `1 s`, and `3 s` occlusions
- recovery success rate by occlusion duration
- recovery success with and without phone support

### 9.4 Recommended Acceptance Targets

For field readiness, target:

- wrong-person lock rate `< 0.1%` of evaluated frames
- zero direct person-to-person target switches without an intervening `lost`
- median recovery `< 1.5 s` after short occlusion
- robust lock retention through `<= 1.0 s` partial occlusions in moderate crowds

## 10. Explicit Limits

This system will not reliably solve all cases.

Known limits:

- monocular video alone cannot guarantee safe relock after long full occlusion in dense crowds
- face verification becomes unreliable in low light, profile views, and motion blur
- clothing changes after enrollment reduce body re-ID utility until face or phone support returns
- world position from monocular geometry is weak when feet are hidden or camera pose is uncertain

Safe behavior under these limits is:

- keep current lock only while continuity is credible
- otherwise degrade to `weak`
- then declare `lost`
- request help instead of guessing

## Summary

The recommended system is a conservative multi-cue tracker:

- person detector plus multi-track manager for short-term continuity
- body re-ID and pose for long-term identity maintenance
- face verification only for lock and relock
- phone signal as optional support
- explicit lock-state hysteresis and confidence decay
- strong anti-switch rules for crowds and occlusion

If implemented this way, the drone will sometimes lose the journalist, but it will be much less likely to follow the wrong person.
