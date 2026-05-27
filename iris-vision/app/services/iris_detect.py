"""MediaPipe 眼部定位与虹膜环带 mask 生成。"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from app.services.face_landmarker import detect_face_landmarks

# 虹膜中心 landmark（Face Landmarker 全量模型）
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


def _landmark_to_pixel(landmark, width: int, height: int) -> Tuple[int, int]:
    """将 MediaPipe 归一化坐标转为像素坐标。"""
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    return x, y


def _estimate_iris_radius(landmarks, center_idx: int, width: int, height: int) -> float:
    """
    用虹膜周围若干 landmark 到中心的平均距离估计半径。
    """
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


def detect_iris_ring_mask(
    image_bgr: np.ndarray,
    inner_ratio: float = 0.30,
    outer_ratio: float = 0.80,
) -> Optional[IrisDetectionResult]:
    """
    检测虹膜并生成环带 mask（排除瞳孔中心和高光区域外的巩膜）。

    优先使用右眼（与 Pan 2017 一致）；若右眼不可用则尝试左眼。
    """
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    landmarks = detect_face_landmarks(rgb)

    if landmarks is None:
        return None

    candidates = [
        (_RIGHT_IRIS_CENTER, "right"),
        (_LEFT_IRIS_CENTER, "left"),
    ]

    for center_idx, side in candidates:
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
            radius=radius,
            eye_side=side,
            sample_pixel_count=sample_count,
        )

    return None
