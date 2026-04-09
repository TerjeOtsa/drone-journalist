"""
Live camera tracker for bench-testing the perception stack.

This runner is intentionally conservative:

- user manually enrolls a person with an ROI
- optical flow + template matching maintain short-term track continuity
- optional HOG detections assist relock and wrong-person avoidance
- results are fed through the shared identity/lock manager

Usage examples:
    python -m perception.live_camera --camera 0
    python -m perception.live_camera --video path\\to\\clip.mp4
    python -m perception.live_camera --camera 0 --log live_tracks.ndjson
    python -m perception.live_camera --camera 0 --roi 400,180,160,320
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from interfaces.schemas import LockState
from perception.geometry import (MonocularGroundProjector,
                                 MonocularProjectorParams)
from perception.identity_lock import IdentityLockManager
from perception.parameters import IdentityLockParams
from perception.schemas import (CandidateObservation, IdentityCues,
                                ImageTarget, TrackingCues)

LOCK_COLORS = {
    LockState.CANDIDATE: (0, 191, 255),
    LockState.LOCKED: (0, 200, 120),
    LockState.WEAK: (0, 215, 255),
    LockState.LOST: (0, 80, 220),
}


@dataclass
class LiveTrackerParams:
    """Tunable knobs for the OpenCV-based live person tracker."""

    max_corners: int = 80
    min_corners: int = 12
    feature_quality: float = 0.01
    min_corner_distance: float = 5.0
    flow_win_size: int = 21
    refresh_features_every: int = 8
    template_search_scale: float = 2.0
    template_match_threshold: float = 0.40
    relock_template_threshold: float = 0.50
    hog_stride: tuple[int, int] = (8, 8)
    hog_padding: tuple[int, int] = (16, 16)
    hog_scale: float = 1.05
    detector_interval: int = 10
    blur_reference: float = 180.0
    min_bbox_area_px: float = 1200.0
    roi_margin_px: int = 20
    body_hist_bins: int = 16
    target_id: str = "subject_primary"
    projector: MonocularProjectorParams = field(default_factory=MonocularProjectorParams)
    # Face detection
    face_detect_interval: int = 5       # run face detector every N frames
    face_min_size_px: int = 30          # smallest face to detect (pixels)
    face_scale_factor: float = 1.15     # Haar cascade scale factor
    face_min_neighbors: int = 4         # Haar cascade min-neighbors (strictness)
    face_search_expand: float = 0.3     # expand person bbox by this fraction to search for face
    face_score_decay_tau: float = 1.5   # seconds — face score decays when not detected
    face_quality_weight: float = 0.6    # blend of size-score vs sharpness-score


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2]) * max(0.0, bbox[3])


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return bbox[0] + 0.5 * bbox[2], bbox[1] + 0.5 * bbox[3]


def _clip_bbox(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> Optional[tuple[int, int, int, int]]:
    x, y, w, h = bbox
    x1 = int(round(_clamp(x, 0.0, width - 1.0)))
    y1 = int(round(_clamp(y, 0.0, height - 1.0)))
    x2 = int(round(_clamp(x + w, 0.0, width)))
    y2 = int(round(_clamp(y + h, 0.0, height)))
    w2 = max(0, x2 - x1)
    h2 = max(0, y2 - y1)
    if w2 <= 1 or h2 <= 1:
        return None
    return x1, y1, w2, h2


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = _bbox_area(a) + _bbox_area(b) - inter
    if union <= 1e-6:
        return 0.0
    return inter / union


def _extract_patch(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> Optional[np.ndarray]:
    clipped = _clip_bbox(bbox, frame.shape[1], frame.shape[0])
    if clipped is None:
        return None
    x, y, w, h = clipped
    return frame[y:y + h, x:x + w].copy()


def _blur_score(gray: np.ndarray, bbox: tuple[float, float, float, float]) -> float:
    patch = _extract_patch(gray, bbox)
    if patch is None or patch.size == 0:
        return 0.0
    return float(cv2.Laplacian(patch, cv2.CV_64F).var())


def _appearance_histogram(patch_bgr: np.ndarray, bins: int) -> np.ndarray:
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return hist


def _appearance_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dist = cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)
    return float(_clamp(1.0 - dist, 0.0, 1.0))


def _safe_resize_gray(patch: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(patch, size, interpolation=cv2.INTER_LINEAR)


def _detect_faces_in_region(
    gray: np.ndarray,
    bbox: tuple[float, float, float, float],
    cascade_frontal: cv2.CascadeClassifier,
    cascade_profile: cv2.CascadeClassifier,
    params: "LiveTrackerParams",
) -> list[tuple[int, int, int, int]]:
    """Run Haar face detection inside an expanded person bounding box.

    Returns list of (x, y, w, h) face rects in full-frame coordinates.
    """
    h_frame, w_frame = gray.shape[:2]
    bx, by, bw, bh = bbox
    # Expand the search region to catch faces at the edge of the body bbox
    expand = params.face_search_expand
    sx = int(round(bx - bw * expand))
    sy = int(round(by - bh * expand * 0.5))  # mostly upward
    sw = int(round(bw * (1.0 + 2.0 * expand)))
    sh = int(round(bh * (0.5 + expand)))  # upper half of body + margin
    # Clip to frame
    sx = max(0, sx)
    sy = max(0, sy)
    sw = min(sw, w_frame - sx)
    sh = min(sh, h_frame - sy)
    if sw < params.face_min_size_px or sh < params.face_min_size_px:
        return []
    roi = gray[sy : sy + sh, sx : sx + sw]
    min_sz = (params.face_min_size_px, params.face_min_size_px)

    faces: list[tuple[int, int, int, int]] = []
    for cascade in (cascade_frontal, cascade_profile):
        detections = cascade.detectMultiScale(
            roi,
            scaleFactor=params.face_scale_factor,
            minNeighbors=params.face_min_neighbors,
            minSize=min_sz,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        for (fx, fy, fw, fh) in detections:
            faces.append((sx + fx, sy + fy, fw, fh))
    return faces


def _face_score_from_detection(
    face_rect: tuple[int, int, int, int],
    gray: np.ndarray,
    person_bbox: tuple[float, float, float, float],
    params: "LiveTrackerParams",
) -> float:
    """Score a detected face on [0, 1] based on size and sharpness."""
    _, _, fw, fh = face_rect
    face_area = fw * fh
    # Size score: larger face in frame → higher confidence
    # A 60×60px face is "good", 120×120 is "excellent"
    size_score = _clamp(face_area / (80.0 * 80.0), 0.0, 1.0)
    # Sharpness score: Laplacian variance on the face patch
    fx, fy = face_rect[0], face_rect[1]
    face_patch = gray[fy : fy + fh, fx : fx + fw]
    if face_patch.size == 0:
        return size_score
    sharpness = float(cv2.Laplacian(face_patch, cv2.CV_64F).var())
    sharpness_score = _clamp(sharpness / max(params.blur_reference, 1e-6), 0.0, 1.0)
    w = params.face_quality_weight
    return w * size_score + (1.0 - w) * sharpness_score


class LiveCameraTracker:
    """OpenCV-based live person tracker with manual enrollment."""

    def __init__(
        self,
        params: LiveTrackerParams | None = None,
        lock_params: IdentityLockParams | None = None,
    ) -> None:
        self.p = params or LiveTrackerParams()
        self.lock_params = lock_params or IdentityLockParams(
            candidate_stable_time_s=0.20,
            relock_stable_time_s=0.20,
            weak_to_lost_timeout_s=1.0,
            candidate_help_after_s=3.0,
            weak_help_after_s=1.5,
            lost_help_after_s=2.0,
            image_only_tracking_cap=0.62,
        )
        self.projector = MonocularGroundProjector(self.p.projector)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())  # type: ignore[attr-defined]
        # Face detection cascades (ship with every OpenCV install)
        self._face_cascade_frontal = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"  # type: ignore[attr-defined]
        )
        self._face_cascade_profile = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"  # type: ignore[attr-defined]
        )
        self.reset()

    def reset(self) -> None:
        """Clear all tracking state and start fresh."""
        self.lock_manager = IdentityLockManager(self.lock_params)
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None
        self.current_bbox: Optional[tuple[float, float, float, float]] = None
        self.template_gray: Optional[np.ndarray] = None
        self.template_hist: Optional[np.ndarray] = None
        self.frame_id = 0
        self.last_hog_frame = -10_000
        self.last_hog_candidates: list[tuple[tuple[float, float, float, float], float]] = []
        self.last_timestamp: Optional[float] = None
        # Face detection state
        self._last_face_frame = -10_000
        self._face_score: float = 0.0           # current smoothed face score [0,1]
        self._face_visible: bool = False         # was a face detected on the last check?
        self._last_face_detect_ts: Optional[float] = None
        self._face_bbox: Optional[tuple[int, int, int, int]] = None  # last detected face rect

    def initialize_target(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        """Enroll a target from *frame* inside *bbox* (x, y, w, h)."""
        x, y, w, h = bbox
        self.current_bbox = (float(x), float(y), float(w), float(h))
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._refresh_templates(frame, self.current_bbox)
        self._refresh_points(self.prev_gray, self.current_bbox)
        self.lock_manager = IdentityLockManager(self.lock_params)
        self.last_timestamp = None

    def process(self, frame: np.ndarray, timestamp: float) -> tuple[object, dict]:
        """Run one tracking cycle and return ``(lock_result, debug_dict)``."""
        self.frame_id += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        tracker_debug: dict[str, object] = {
            "flow_ratio": 0.0,
            "template_score": 0.0,
            "hog_score": 0.0,
            "bbox_px": None,
        }

        if self.current_bbox is None or self.template_gray is None or self.template_hist is None:
            obs = CandidateObservation(
                timestamp=timestamp,
                frame_id=self.frame_id,
                candidate_id=None,
                image_target=None,
                world_target=None,
                identity=IdentityCues(),
                tracking=TrackingCues(),
                best_candidate_score=0.0,
                second_best_score=0.0,
                has_fresh_detection=False,
                failure_reason="not_enrolled",
            )
            result = self.lock_manager.update(obs)
            self.prev_gray = gray
            self.last_timestamp = timestamp
            return result, tracker_debug

        flow_bbox, flow_ratio = self._track_by_flow(gray)
        predicted_bbox = flow_bbox or self.current_bbox
        template_bbox, template_score = self._track_by_template(gray, predicted_bbox)

        use_hog = (
            self.frame_id - self.last_hog_frame >= self.p.detector_interval
            or self.lock_manager.lock_state in (LockState.WEAK, LockState.LOST)
        )
        hog_best_bbox = None
        hog_best_score = 0.0
        second_best_score = 0.0
        if use_hog:
            hog_candidates = self._run_hog(frame, predicted_bbox)
            self.last_hog_candidates = hog_candidates
            self.last_hog_frame = self.frame_id
        else:
            hog_candidates = self.last_hog_candidates

        if hog_candidates:
            hog_best_bbox, hog_best_score = hog_candidates[0]
            if len(hog_candidates) > 1:
                second_best_score = hog_candidates[1][1]

        fused_bbox = None
        if hog_best_bbox is not None and hog_best_score >= 0.55:
            if template_score >= self.p.template_match_threshold:
                fused_bbox = self._blend_bbox(hog_best_bbox, template_bbox, 0.65)
            else:
                fused_bbox = hog_best_bbox
        elif template_score >= self.p.template_match_threshold:
            fused_bbox = template_bbox
        elif flow_bbox is not None and flow_ratio >= 0.45:
            fused_bbox = flow_bbox

        image_target = None
        world_target = None
        body_similarity = 0.0
        visibility = 0.0
        quality = 0.0
        motion_cue = 0.0
        association = 0.0
        best_candidate_score = 0.0
        failure_reason = None

        if fused_bbox is not None:
            fused_bbox = self._sanitize_bbox(fused_bbox, width, height)
            tracker_debug["bbox_px"] = tuple(int(round(v)) for v in fused_bbox)

        if fused_bbox is not None:
            patch = _extract_patch(frame, fused_bbox)
            patch_gray = _extract_patch(gray, fused_bbox)
            if patch is not None and patch_gray is not None and _bbox_area(fused_bbox) >= self.p.min_bbox_area_px:
                hist = _appearance_histogram(patch, self.p.body_hist_bins)
                body_similarity = _appearance_similarity(hist, self.template_hist)

                x, y, w, h = fused_bbox
                image_target = ImageTarget(
                    cx_norm=(x + 0.5 * w) / width,
                    cy_norm=(y + 0.5 * h) / height,
                    w_norm=w / width,
                    h_norm=h / height,
                    footpoint_norm=((x + 0.5 * w) / width, (y + h) / height),
                    velocity_px_s=self._bbox_velocity_px_s(fused_bbox, timestamp),
                    bbox_confidence=float(_clamp(max(template_score, hog_best_score, flow_ratio), 0.0, 1.0)),
                )
                world_target = self.projector.project(image_target)

                visibility = float(_clamp(self._visible_fraction(fused_bbox, width, height), 0.0, 1.0))
                quality = float(_clamp(_blur_score(gray, fused_bbox) / max(self.p.blur_reference, 1e-6), 0.0, 1.0))
                association = max(flow_ratio, _iou(predicted_bbox, fused_bbox))
                motion_cue = self._motion_consistency(predicted_bbox, fused_bbox)
                best_candidate_score = max(
                    0.45 * body_similarity +
                    0.25 * template_score +
                    0.20 * association +
                    0.10 * visibility,
                    hog_best_score,
                )

                if body_similarity >= 0.68 or template_score >= self.p.relock_template_threshold:
                    self.current_bbox = fused_bbox
                    self._refresh_points(gray, fused_bbox)
                    if (
                        self.lock_manager.lock_state == LockState.LOCKED
                        and quality >= 0.55
                        and body_similarity >= 0.72
                    ):
                        self._refresh_templates(frame, fused_bbox, alpha=0.15)
            else:
                failure_reason = "bbox_too_small"
        else:
            failure_reason = "tracker_unresolved"

        if fused_bbox is None:
            self.prev_points = None

        tracker_debug["flow_ratio"] = round(flow_ratio, 3)
        tracker_debug["template_score"] = round(template_score, 3)
        tracker_debug["hog_score"] = round(hog_best_score, 3)

        # ── Face detection (opportunistic, every N frames) ───────────
        face_cue: Optional[float] = None
        face_used = False
        if fused_bbox is not None and _bbox_area(fused_bbox) >= self.p.min_bbox_area_px:
            run_face = (
                self.frame_id - self._last_face_frame >= self.p.face_detect_interval
                or self.lock_manager.lock_state in (LockState.CANDIDATE, LockState.WEAK)
            )
            if run_face:
                self._last_face_frame = self.frame_id
                faces = _detect_faces_in_region(
                    gray, fused_bbox,
                    self._face_cascade_frontal,
                    self._face_cascade_profile,
                    self.p,
                )
                if faces:
                    # Pick the largest detected face
                    best_face = max(faces, key=lambda f: f[2] * f[3])
                    self._face_bbox = best_face
                    self._face_score = _face_score_from_detection(
                        best_face, gray, fused_bbox, self.p,
                    )
                    self._face_visible = True
                    self._last_face_detect_ts = timestamp
                    face_cue = float(_clamp(self._face_score, 0.0, 1.0))
                    face_used = True
                else:
                    self._face_visible = False
                    self._face_bbox = None
                    # Decay the face score when not detected
                    if self._last_face_detect_ts is not None:
                        import math as _math
                        dt_face = max(0.0, timestamp - self._last_face_detect_ts)
                        self._face_score *= _math.exp(-dt_face / max(self.p.face_score_decay_tau, 1e-6))
                    face_cue = float(_clamp(self._face_score, 0.0, 1.0)) if self._face_score > 0.05 else None
        else:
            self._face_visible = False
            if self._last_face_detect_ts is not None and self.last_timestamp is not None:
                import math as _math
                dt_face = max(0.0, timestamp - self._last_face_detect_ts)
                self._face_score *= _math.exp(-dt_face / max(self.p.face_score_decay_tau, 1e-6))

        tracker_debug["face_score"] = round(self._face_score, 3)
        tracker_debug["face_visible"] = self._face_visible
        tracker_debug["face_bbox"] = self._face_bbox if self._face_visible else None

        has_detection = fused_bbox is not None
        obs = CandidateObservation(
            timestamp=timestamp,
            frame_id=self.frame_id,
            candidate_id=self.p.target_id if has_detection else None,
            track_id=f"{self.p.target_id}:cam",
            image_target=image_target,
            world_target=world_target,
            identity=IdentityCues(
                body=body_similarity if has_detection else None,
                pose=None,
                phone=None,
                face=face_cue,
            ),
            tracking=TrackingCues(
                detection=max(template_score, hog_best_score) if has_detection else 0.0,
                association=association if has_detection else 0.0,
                motion=motion_cue if has_detection else 0.0,
                visibility=visibility if has_detection else 0.0,
                quality=quality if has_detection else 0.0,
            ),
            best_candidate_score=best_candidate_score,
            second_best_score=second_best_score,
            has_fresh_detection=has_detection,
            blur_score=1.0 - quality if has_detection else 1.0,
            face_used=face_used,
            phone_used=False,
            failure_reason=failure_reason,
        )

        result = self.lock_manager.update(obs)
        self.prev_gray = gray
        self.last_timestamp = timestamp
        return result, tracker_debug

    def _track_by_flow(self, gray: np.ndarray) -> tuple[Optional[tuple[float, float, float, float]], float]:
        if self.prev_gray is None or self.prev_points is None or self.current_bbox is None:
            return None, 0.0
        if len(self.prev_points) < self.p.min_corners:
            return None, 0.0

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_points,
            None,  # type: ignore[arg-type]  # cv2 stubs lack None-accepting overload
            winSize=(self.p.flow_win_size, self.p.flow_win_size),
            maxLevel=3,
        )
        if next_points is None or status is None:
            return None, 0.0

        good_prev = self.prev_points[status.flatten() == 1]
        good_next = next_points[status.flatten() == 1]
        if len(good_next) < self.p.min_corners:
            self.prev_points = None
            return None, len(good_next) / max(len(self.prev_points or []), 1)

        displacement = np.median(good_next - good_prev, axis=0)
        prev_center = np.median(good_prev, axis=0)
        next_center = np.median(good_next, axis=0)
        prev_spread = np.mean(np.linalg.norm(good_prev - prev_center, axis=1))
        next_spread = np.mean(np.linalg.norm(good_next - next_center, axis=1))
        scale = 1.0
        if prev_spread > 1e-3:
            scale = float(_clamp(next_spread / prev_spread, 0.8, 1.25))

        x, y, w, h = self.current_bbox
        cx, cy = x + 0.5 * w, y + 0.5 * h
        cx += float(displacement[0])
        cy += float(displacement[1])
        new_w = w * scale
        new_h = h * scale
        tracked_bbox = (cx - 0.5 * new_w, cy - 0.5 * new_h, new_w, new_h)

        self.prev_points = good_next.reshape(-1, 1, 2)
        return tracked_bbox, len(good_next) / max(len(status), 1)

    def _track_by_template(
        self,
        gray: np.ndarray,
        predicted_bbox: tuple[float, float, float, float],
    ) -> tuple[tuple[float, float, float, float], float]:
        assert self.template_gray is not None
        x, y, w, h = predicted_bbox
        search_w = max(int(round(w * self.p.template_search_scale)), self.template_gray.shape[1] + 4)
        search_h = max(int(round(h * self.p.template_search_scale)), self.template_gray.shape[0] + 4)
        cx, cy = _bbox_center(predicted_bbox)

        sx = int(round(cx - 0.5 * search_w))
        sy = int(round(cy - 0.5 * search_h))
        search_bbox = (sx, sy, search_w, search_h)
        search_patch = _extract_patch(gray, search_bbox)
        clipped = _clip_bbox(search_bbox, gray.shape[1], gray.shape[0])
        if search_patch is None or clipped is None:
            return predicted_bbox, 0.0

        tpl = self.template_gray
        if search_patch.shape[0] <= tpl.shape[0] or search_patch.shape[1] <= tpl.shape[1]:
            return predicted_bbox, 0.0

        match = cv2.matchTemplate(search_patch, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(match)
        cx0, cy0, _, _ = clipped
        tx = cx0 + max_loc[0]
        ty = cy0 + max_loc[1]
        tw = tpl.shape[1]
        th = tpl.shape[0]
        return (float(tx), float(ty), float(tw), float(th)), float(_clamp(max_val, 0.0, 1.0))

    def _run_hog(
        self,
        frame: np.ndarray,
        predicted_bbox: tuple[float, float, float, float],
    ) -> list[tuple[tuple[float, float, float, float], float]]:
        rects, _weights = self.hog.detectMultiScale(
            frame,
            winStride=self.p.hog_stride,
            padding=self.p.hog_padding,
            scale=self.p.hog_scale,
        )

        candidates: list[tuple[tuple[float, float, float, float], float]] = []
        pred_cx, pred_cy = _bbox_center(predicted_bbox)
        pred_diag = max((_bbox_area(predicted_bbox) ** 0.5), 1.0)
        for rect in rects:
            bbox: tuple[float, float, float, float] = (
                float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]),
            )
            patch = _extract_patch(frame, bbox)
            if patch is None or self.template_hist is None:
                continue
            hist = _appearance_histogram(patch, self.p.body_hist_bins)
            appearance = _appearance_similarity(hist, self.template_hist)
            det_cx, det_cy = _bbox_center(bbox)
            dist = ((det_cx - pred_cx) ** 2 + (det_cy - pred_cy) ** 2) ** 0.5
            proximity = _clamp(1.0 - dist / (2.5 * pred_diag), 0.0, 1.0)
            overlap = _iou(bbox, predicted_bbox)
            score = 0.50 * appearance + 0.25 * proximity + 0.25 * overlap
            candidates.append((bbox, float(score)))

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:4]

    def _refresh_points(self, gray: np.ndarray, bbox: tuple[float, float, float, float]) -> None:
        mask = np.zeros_like(gray)
        clipped = _clip_bbox(bbox, gray.shape[1], gray.shape[0])
        if clipped is None:
            self.prev_points = None
            return
        x, y, w, h = clipped
        margin = min(self.p.roi_margin_px, max(1, min(w, h) // 6))
        x1 = x + margin
        y1 = y + margin
        x2 = max(x1 + 1, x + w - margin)
        y2 = max(y1 + 1, y + h - margin)
        mask[y1:y2, x1:x2] = 255
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.p.max_corners,
            qualityLevel=self.p.feature_quality,
            minDistance=self.p.min_corner_distance,
            mask=mask,
        )
        self.prev_points = points

    def _refresh_templates(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        alpha: float = 1.0,
    ) -> None:
        patch = _extract_patch(frame, bbox)
        if patch is None or patch.size == 0:
            return
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        hist = _appearance_histogram(patch, self.p.body_hist_bins)
        if alpha >= 1.0 or self.template_gray is None or self.template_hist is None:
            self.template_gray = gray
            self.template_hist = hist
            return

        resized_prev = _safe_resize_gray(self.template_gray, (gray.shape[1], gray.shape[0]))
        blended = cv2.addWeighted(resized_prev, 1.0 - alpha, gray, alpha, 0.0)
        self.template_gray = blended
        assert self.template_hist is not None  # guarded by early return above
        blended_hist = (1.0 - alpha) * self.template_hist + alpha * hist
        total = float(np.sum(blended_hist))
        if total > 1e-6:
            blended_hist = blended_hist / total
        self.template_hist = blended_hist

    def _sanitize_bbox(
        self,
        bbox: tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> tuple[float, float, float, float]:
        clipped = _clip_bbox(bbox, width, height)
        if clipped is None:
            return bbox
        x, y, w, h = clipped
        return float(x), float(y), float(w), float(h)

    def _blend_bbox(
        self,
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
        weight_a: float,
    ) -> tuple[float, float, float, float]:
        wa = _clamp(weight_a, 0.0, 1.0)
        wb = 1.0 - wa
        return (
            wa * a[0] + wb * b[0],
            wa * a[1] + wb * b[1],
            wa * a[2] + wb * b[2],
            wa * a[3] + wb * b[3],
        )

    def _visible_fraction(self, bbox: tuple[float, float, float, float], width: int, height: int) -> float:
        clipped = _clip_bbox(bbox, width, height)
        if clipped is None:
            return 0.0
        return _bbox_area(clipped) / max(_bbox_area(bbox), 1e-6)

    def _motion_consistency(
        self,
        predicted_bbox: tuple[float, float, float, float],
        fused_bbox: tuple[float, float, float, float],
    ) -> float:
        px, py = _bbox_center(predicted_bbox)
        fx, fy = _bbox_center(fused_bbox)
        jump = ((fx - px) ** 2 + (fy - py) ** 2) ** 0.5
        scale = max((_bbox_area(predicted_bbox) ** 0.5), 1.0)
        return float(_clamp(1.0 - jump / (0.75 * scale), 0.0, 1.0))

    def _bbox_velocity_px_s(
        self,
        bbox: tuple[float, float, float, float],
        timestamp: float,
    ) -> tuple[float, float]:
        if self.current_bbox is None or self.last_timestamp is None:
            return (0.0, 0.0)
        dt = max(timestamp - self.last_timestamp, 1e-6)
        prev_cx, prev_cy = _bbox_center(self.current_bbox)
        cx, cy = _bbox_center(bbox)
        return ((cx - prev_cx) / dt, (cy - prev_cy) / dt)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bench-test the perception stack on a live camera")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=0, help="Camera index to open")
    src.add_argument("--video", type=str, default=None, help="Optional video file instead of a live camera")
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height")
    parser.add_argument("--fps", type=float, default=30.0, help="Requested capture FPS")
    parser.add_argument("--log", type=Path, default=None, help="Optional NDJSON log path")
    parser.add_argument("--roi", type=str, default=None, help="Initial ROI as x,y,w,h")
    parser.add_argument("--no-world", action="store_true", help="Disable monocular world estimation")
    return parser.parse_args()


def _open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        raise RuntimeError("Could not open the requested camera/video source")
    return cap


def _parse_roi(value: str) -> tuple[int, int, int, int]:
    parts = [int(v.strip()) for v in value.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be x,y,w,h")
    return parts[0], parts[1], parts[2], parts[3]


def _draw_overlay(
    frame: np.ndarray,
    result,
    debug: dict,
    fps: float,
) -> np.ndarray:
    canvas = frame.copy()
    color = LOCK_COLORS[result.lock_state]

    image = result.target_position_image
    if image is not None:
        h, w = canvas.shape[:2]
        x = int(round((image.cx_norm - 0.5 * image.w_norm) * w))
        y = int(round((image.cy_norm - 0.5 * image.h_norm) * h))
        bw = int(round(image.w_norm * w))
        bh = int(round(image.h_norm * h))
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color, 2)
        foot = image.footpoint_norm
        if foot is not None:
            fx = int(round(foot[0] * w))
            fy = int(round(foot[1] * h))
            cv2.circle(canvas, (fx, fy), 4, color, -1)

    y0 = 24
    for line in [
        f"lock={result.lock_state.value}  track={result.tracking_confidence:.2f}  id={result.identity_confidence:.2f}  fps={fps:.1f}",
        f"flow={debug['flow_ratio']:.2f}  tpl={debug['template_score']:.2f}  hog={debug['hog_score']:.2f}  face={debug.get('face_score', 0.0):.2f}",
        "keys: l=select target  r=reset  q=quit",
    ]:
        cv2.putText(canvas, line, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y0 += 26

    # Draw face bbox if available
    face_bbox = debug.get("face_bbox")
    if face_bbox is not None:
        fx, fy, fw, fh = face_bbox
        face_color = (255, 200, 0)  # cyan-ish for face
        cv2.rectangle(canvas, (fx, fy), (fx + fw, fy + fh), face_color, 1)
        cv2.putText(canvas, "face", (fx, fy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, face_color, 1, cv2.LINE_AA)

    world = result.target_position_world
    if world is not None:
        cv2.putText(
            canvas,
            f"world[{world.frame}] = ({world.position_m.x:.1f}, {world.position_m.y:.1f}, {world.position_m.z:.1f}) m",
            (12, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        y0 += 24

    if result.diagnostics.failure_reason:
        cv2.putText(
            canvas,
            f"reason: {result.diagnostics.failure_reason}",
            (12, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 255),
            2,
            cv2.LINE_AA,
        )

    return canvas


def _log_result(handle, result) -> None:
    handle.write(json.dumps(result.to_dict()) + "\n")
    handle.flush()


def main() -> None:
    args = _parse_args()
    cap = _open_capture(args)

    params = LiveTrackerParams(
        projector=MonocularProjectorParams(
            image_width=args.width,
            image_height=args.height,
        )
    )
    tracker = LiveCameraTracker(params=params)
    if args.no_world:
        tracker.projector = MonocularGroundProjector(
            MonocularProjectorParams(
                image_width=args.width,
                image_height=args.height,
                camera_height_m=0.0,
                min_depression_deg=90.0,
            )
        )

    log_handle = None
    if args.log is not None:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log_handle = args.log.open("w", encoding="utf-8")

    fps = 0.0
    prev_t = time.time()

    try:
        initialized_from_cli = False
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            timestamp = time.time()
            dt = max(timestamp - prev_t, 1e-6)
            prev_t = timestamp
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0.0 else 1.0 / dt

            if args.roi and not initialized_from_cli:
                tracker.initialize_target(frame, _parse_roi(args.roi))
                initialized_from_cli = True

            result, debug = tracker.process(frame, timestamp)
            if log_handle is not None:
                _log_result(log_handle, result)

            view = _draw_overlay(frame, result, debug, fps)
            cv2.imshow("Live Camera Tracker", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                tracker.reset()
                initialized_from_cli = False
            if key == ord("l"):
                roi = cv2.selectROI("Live Camera Tracker", frame, fromCenter=False, showCrosshair=True)
                if roi and roi[2] > 0 and roi[3] > 0:
                    tracker.initialize_target(
                        frame, (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])),
                    )

    finally:
        if log_handle is not None:
            log_handle.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
