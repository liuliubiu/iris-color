"""虹膜环带 mask 生成：支持眼部特写与全脸两种模式。"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from app.services.eye_iris_detect import build_iris_ring_from_pupil, detect_pupil
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
    landmarks = detect_face_landmarks(rgb)
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
        )
    return None


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

    mask, outer_r, inner_r, sample_count = build_iris_ring_from_pupil(
        image_bgr.shape,
        pupil,
        inner_pupil_multiplier=eye_cfg.get("inner_pupil_multiplier", 1.15),
        outer_pupil_multiplier=eye_cfg.get("outer_pupil_multiplier", 2.8),
    )
    return IrisDetectionResult(
        mask=mask,
        center=(pupil.center_x, pupil.center_y),
        radius=outer_r,
        eye_side="closeup",
        sample_pixel_count=sample_count,
        method="eye_closeup",
        pupil_center=(pupil.center_x, pupil.center_y),
        pupil_radius=pupil.radius,
        inner_radius=inner_r,
        outer_radius=outer_r,
    )


def detect_iris_ring_mask(
    image_bgr: np.ndarray,
    mode: str = "eye_closeup",
    inner_ratio: float = 0.30,
    outer_ratio: float = 0.80,
    eye_closeup_cfg: Optional[dict] = None,
) -> Optional[IrisDetectionResult]:
    """
    检测虹膜环带 mask。

    mode:
      - eye_closeup: 默认，适用于「眼睛占满画面」的拍照/上传图
      - face: 全脸图，MediaPipe 定位
      - auto: 先 eye_closeup，失败再 face
    """
    eye_cfg = eye_closeup_cfg or {}

    if mode == "eye_closeup":
        return _detect_from_eye_closeup(image_bgr, eye_cfg)
    if mode == "face":
        return _detect_from_face_landmarks(image_bgr, inner_ratio, outer_ratio)

    # auto
    result = _detect_from_eye_closeup(image_bgr, eye_cfg)
    if result is not None:
        return result
    return _detect_from_face_landmarks(image_bgr, inner_ratio, outer_ratio)
