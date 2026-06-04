"""虹膜环带 mask 生成：支持眼部特写与全脸两种模式。"""

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import cv2
import numpy as np

from app.services.eye_iris_detect import build_refined_iris_ring, detect_pupil, detect_rough_iris_disk
from app.services.face_landmarker import detect_face_landmarks

_LEFT_IRIS_CENTER = 468
_RIGHT_IRIS_CENTER = 473


@dataclass
class IrisDetectionResult:
    """虹膜检测结果。"""

    mask: np.ndarray
    center: Tuple[int, int]
    radius: float
    eye_side: str
    sample_pixel_count: int
    method: str = "eye_closeup"
    pupil_center: Optional[Tuple[int, int]] = None
    pupil_radius: Optional[float] = None
    inner_radius: Optional[float] = None
    outer_radius: Optional[float] = None
    pupil_confidence: Optional[float] = None
    iris_confidence: Optional[float] = None
    pupil_method: Optional[str] = None
    iris_outer_method: Optional[str] = None
    candidate_count: Optional[int] = None
    candidate_mask: Optional[np.ndarray] = None
    highlight_mask: Optional[np.ndarray] = None
    exclusion_mask: Optional[np.ndarray] = None
    selection_score: Optional[float] = None


def build_manual_iris_detection(
    image_shape: Tuple[int, int],
    params: Mapping[str, float],
) -> Optional[IrisDetectionResult]:
    """按人工调整参数直接生成虹膜环带 mask。"""
    h, w = image_shape[:2]
    min_dim = min(h, w)

    try:
        cx = float(params["center_x"])
        cy = float(params["center_y"])
        pupil_r = float(params["pupil_radius"])
        inner_r = float(params["inner_radius"])
        outer_r = float(params["outer_radius"])
    except (KeyError, TypeError, ValueError):
        return None

    values = [cx, cy, pupil_r, inner_r, outer_r]
    if not all(np.isfinite(values)):
        return None

    cx = float(np.clip(cx, 0, w - 1))
    cy = float(np.clip(cy, 0, h - 1))
    pupil_r = float(np.clip(pupil_r, 2.0, min_dim * 0.45))
    inner_r = float(np.clip(inner_r, pupil_r + 1.0, min_dim * 0.48))
    outer_r = float(np.clip(outer_r, inner_r + 2.0, min_dim * 0.50))
    if outer_r <= inner_r + 2:
        return None

    center = (int(round(cx)), int(round(cy)))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, int(round(outer_r)), 255, -1)
    cv2.circle(mask, center, int(round(inner_r)), 0, -1)
    sample_count = int(np.count_nonzero(mask))
    if sample_count <= 0:
        return None

    return IrisDetectionResult(
        mask=mask,
        center=center,
        radius=outer_r,
        eye_side="manual",
        sample_pixel_count=sample_count,
        method="manual_adjustment",
        pupil_center=center,
        pupil_radius=pupil_r,
        inner_radius=inner_r,
        outer_radius=outer_r,
        pupil_confidence=1.0,
        iris_confidence=1.0,
        pupil_method="manual_adjustment",
        iris_outer_method="manual_adjustment",
        candidate_count=0,
    )


def _landmark_to_pixel(landmark, width: int, height: int) -> Tuple[int, int]:
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    return x, y


def _estimate_iris_radius(landmarks, center_idx: int, width: int, height: int) -> float:
    center = landmarks[center_idx]
    cx, cy = center.x * width, center.y * height
    iris_edge_indices = {
        _LEFT_IRIS_CENTER: [469, 470, 471, 472],
        _RIGHT_IRIS_CENTER: [474, 475, 476, 477],
    }
    edge_indices = iris_edge_indices.get(center_idx, [])
    if not edge_indices:
        return min(width, height) * 0.04

    distances = []
    for idx in edge_indices:
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        dx = lm.x * width - cx
        dy = lm.y * height - cy
        distances.append((dx * dx + dy * dy) ** 0.5)

    if not distances:
        return min(width, height) * 0.04
    return float(np.mean(distances))


def _detect_from_face_landmarks(
    image_bgr: np.ndarray,
    inner_ratio: float,
    outer_ratio: float,
) -> Optional[IrisDetectionResult]:
    """全脸模式：MediaPipe Face Landmarker + 虹膜 landmark。"""
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    try:
        landmarks = detect_face_landmarks(rgb)
    except (FileNotFoundError, RuntimeError, ValueError):
        return None
    if landmarks is None:
        return None

    for center_idx, side in [(_RIGHT_IRIS_CENTER, "right"), (_LEFT_IRIS_CENTER, "left")]:
        if center_idx >= len(landmarks):
            continue

        center_lm = landmarks[center_idx]
        cx, cy = _landmark_to_pixel(center_lm, w, h)
        radius = _estimate_iris_radius(landmarks, center_idx, w, h)
        if radius < 5:
            continue

        inner_r = radius * inner_ratio
        outer_r = radius * outer_ratio
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), int(outer_r), 255, -1)
        cv2.circle(mask, (cx, cy), int(inner_r), 0, -1)
        sample_count = int(np.count_nonzero(mask))
        if sample_count < 10:
            continue

        return IrisDetectionResult(
            mask=mask,
            center=(cx, cy),
            radius=outer_r,
            eye_side=side,
            sample_pixel_count=sample_count,
            method="face_landmark",
            pupil_center=(cx, cy),
            pupil_radius=inner_r,
            inner_radius=inner_r,
            outer_radius=outer_r,
            pupil_confidence=0.7,
            iris_confidence=0.7,
            pupil_method="face_landmark",
            iris_outer_method="face_landmark",
            candidate_count=0,
        )
    return None


def _detect_from_rough_closeup(
    image_bgr: np.ndarray,
    rough_cfg: dict,
) -> Optional[IrisDetectionResult]:
    """
    实拍粗略模式：将瞳孔+深色虹膜作为整体圆盘定位。
    适合低对比、反光较多、普通手机拍摄的单眼图。
    """
    rough = detect_rough_iris_disk(
        image_bgr,
        center_roi_ratio=rough_cfg.get("center_roi_ratio", 0.92),
        dark_percentile=rough_cfg.get("dark_percentile", 32.0),
        min_radius_ratio=rough_cfg.get("min_radius_ratio", 0.045),
        max_radius_ratio=rough_cfg.get("max_radius_ratio", 0.32),
        inner_iris_ratio=rough_cfg.get("inner_iris_ratio", 0.24),
        outer_iris_ratio=rough_cfg.get("outer_iris_ratio", 0.86),
        highlight_v_threshold=rough_cfg.get("highlight_v_threshold", 235),
    )
    if rough is None:
        return None

    center = (rough.center_x, rough.center_y)
    return IrisDetectionResult(
        mask=_ring_mask_from_rough(image_bgr.shape, center, rough.inner_radius, rough.outer_radius),
        center=center,
        radius=rough.outer_radius,
        eye_side="closeup",
        sample_pixel_count=rough.sample_pixel_count,
        method="rough_closeup",
        pupil_center=center,
        pupil_radius=rough.inner_radius,
        inner_radius=rough.inner_radius,
        outer_radius=rough.outer_radius,
        pupil_confidence=rough.confidence,
        iris_confidence=rough.confidence,
        pupil_method="rough_dark_disk",
        iris_outer_method="rough_dark_disk",
        candidate_count=rough.candidate_count,
        candidate_mask=rough.candidate_mask,
        highlight_mask=rough.highlight_mask,
        exclusion_mask=rough.exclusion_mask,
    )


def _ring_mask_from_rough(
    image_shape: Tuple[int, int],
    center: Tuple[int, int],
    inner_radius: float,
    outer_radius: float,
) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, int(round(max(outer_radius, 1))), 255, -1)
    cv2.circle(mask, center, int(round(max(inner_radius, 1))), 0, -1)
    return mask


def _detect_from_eye_closeup(
    image_bgr: np.ndarray,
    eye_cfg: dict,
) -> Optional[IrisDetectionResult]:
    """
    眼部特写模式：画面主体即为眼睛，通过瞳孔定位虹膜环带。
    不依赖全脸检测。
    """
    pupil = detect_pupil(
        image_bgr,
        center_roi_ratio=eye_cfg.get("center_roi_ratio", 0.85),
        dark_percentile=eye_cfg.get("pupil_dark_percentile", 12.0),
    )
    if pupil is None:
        return None

    ring = build_refined_iris_ring(
        image_bgr,
        pupil,
        inner_pupil_multiplier=eye_cfg.get("inner_pupil_multiplier", 1.15),
        outer_pupil_multiplier=eye_cfg.get("outer_pupil_multiplier", 2.8),
        inner_iris_ratio=eye_cfg.get("inner_iris_ratio", 0.35),
        outer_iris_ratio=eye_cfg.get("outer_iris_ratio", 0.85),
    )
    return IrisDetectionResult(
        mask=ring.mask,
        center=(pupil.center_x, pupil.center_y),
        radius=ring.outer_radius,
        eye_side="closeup",
        sample_pixel_count=ring.sample_pixel_count,
        method="eye_closeup",
        pupil_center=(pupil.center_x, pupil.center_y),
        pupil_radius=pupil.radius,
        inner_radius=ring.inner_radius,
        outer_radius=ring.outer_radius,
        pupil_confidence=pupil.confidence,
        iris_confidence=ring.iris_confidence,
        pupil_method=pupil.method,
        iris_outer_method=ring.iris_outer_method,
        candidate_count=pupil.candidate_count,
        candidate_mask=pupil.candidate_mask,
    )


def _score_detection_candidate(image_bgr: np.ndarray, detection: IrisDetectionResult) -> float:
    """给 auto 模式候选打分，避免第一条成功但明显偏差时直接返回。"""
    h, w = image_bgr.shape[:2]
    min_dim = min(h, w)
    cx, cy = detection.center
    center_dist = (((cx - w / 2) ** 2 + (cy - h / 2) ** 2) ** 0.5) / max(min_dim, 1)
    confidence_values = [
        value for value in (detection.pupil_confidence, detection.iris_confidence) if value is not None
    ]
    confidence = float(np.mean(confidence_values)) if confidence_values else 0.35

    ring = detection.mask > 0
    ring_count = int(np.count_nonzero(ring))
    if ring_count <= 0:
        return -float("inf")
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    rejected_ratio = float(np.mean((value[ring] >= 235) | (value[ring] <= 8)))
    sclera_leak = float(np.mean((value[ring] >= 170) & (saturation[ring] <= 70)))
    radius = float(detection.outer_radius or detection.radius or 0)
    radius_score = 1.0 - min(abs(radius - min_dim * 0.14) / max(min_dim * 0.16, 1.0), 1.0)
    sample_score = min(ring_count / max(min_dim * min_dim * 0.018, 1.0), 1.0)
    method_bonus = {
        "eye_closeup": 0.12,
        "rough_closeup": 0.18,
        "face_landmark": 0.06,
    }.get(detection.method, 0.0)

    score = (
        confidence * 1.55
        + sample_score * 0.55
        + radius_score * 0.35
        + method_bonus
        - center_dist * 0.55
        - rejected_ratio * 0.85
        - sclera_leak * 0.55
    )
    detection.selection_score = float(score)
    return float(score)


def _detect_auto(
    image_bgr: np.ndarray,
    inner_ratio: float,
    outer_ratio: float,
    eye_closeup_cfg: dict,
    rough_closeup_cfg: dict,
) -> Optional[IrisDetectionResult]:
    candidates = [
        _detect_from_eye_closeup(image_bgr, eye_closeup_cfg),
        _detect_from_rough_closeup(image_bgr, rough_closeup_cfg),
        _detect_from_face_landmarks(image_bgr, inner_ratio, outer_ratio),
    ]
    valid = [candidate for candidate in candidates if candidate is not None]
    if not valid:
        return None
    return max(valid, key=lambda item: _score_detection_candidate(image_bgr, item))


def detect_iris_ring_mask(
    image_bgr: np.ndarray,
    mode: str = "eye_closeup",
    inner_ratio: float = 0.30,
    outer_ratio: float = 0.80,
    eye_closeup_cfg: Optional[dict] = None,
    rough_closeup_cfg: Optional[dict] = None,
) -> Optional[IrisDetectionResult]:
    """
    检测虹膜环带 mask。

    mode:
      - eye_closeup: 默认，适用于「眼睛占满画面」的拍照/上传图
      - rough_closeup: 实拍粗略模式，适用于低对比深色虹膜
      - face: 全脸图，MediaPipe 定位
      - auto: 多候选评分选择
    """
    eye_cfg = eye_closeup_cfg or {}
    rough_cfg = rough_closeup_cfg or {}

    if mode == "eye_closeup":
        return _detect_from_eye_closeup(image_bgr, eye_cfg)
    if mode == "rough_closeup":
        return _detect_from_rough_closeup(image_bgr, rough_cfg)
    if mode == "face":
        return _detect_from_face_landmarks(image_bgr, inner_ratio, outer_ratio)

    # auto
    return _detect_auto(image_bgr, inner_ratio, outer_ratio, eye_cfg, rough_cfg)
