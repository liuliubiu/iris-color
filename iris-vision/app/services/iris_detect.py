"""虹膜环带 mask 生成：支持眼部特写与全脸两种模式。"""

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import cv2
import numpy as np

from app.services.eye_iris_detect import (
    build_refined_iris_ring,
    build_ring_from_disk,
    detect_iris_disk,
    detect_pupil,
    suppress_specular,
)
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
            pupil_confidence=0.7,
            iris_confidence=0.7,
            pupil_method="face_landmark",
            iris_outer_method="face_landmark",
            candidate_count=0,
        )
    return None


def _detect_precise(
    detect_image: np.ndarray,
    full_shape: Tuple[int, int],
    eye_cfg: dict,
) -> Optional[IrisDetectionResult]:
    """阶段A：瞳孔精定位 + 径向梯度虹膜外缘（适合清晰分离图）。"""
    pupil = detect_pupil(
        detect_image,
        center_roi_ratio=eye_cfg.get("center_roi_ratio", 0.85),
        dark_percentile=eye_cfg.get("pupil_dark_percentile", 12.0),
    )
    if pupil is None or pupil.radius <= 0:
        return None

    ring = build_refined_iris_ring(
        detect_image,
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
        method="eye_closeup_precise",
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


def _detect_rough(
    detect_image: np.ndarray,
    full_shape: Tuple[int, int],
    eye_cfg: dict,
) -> Optional[IrisDetectionResult]:
    """阶段B：黑盘直检（适合瞳孔与虹膜融成一团黑的实拍图）。"""
    disk_cfg = eye_cfg.get("disk", {})
    disk = detect_iris_disk(
        detect_image,
        dark_percentile=disk_cfg.get("dark_percentile", 35.0),
        min_radius_ratio=disk_cfg.get("min_radius_ratio", 0.12),
        max_radius_ratio=disk_cfg.get("max_radius_ratio", 0.45),
        circularity_min=disk_cfg.get("circularity_min", 0.55),
        surround_contrast=disk_cfg.get("surround_contrast", 8.0),
        surround_support_min=disk_cfg.get("surround_support_min", 0.45),
        pupil_iris_ratio=eye_cfg.get("pupil_iris_ratio", 0.30),
        sclera_min=disk_cfg.get("sclera_min", 0.10),
        sclera_v_min=disk_cfg.get("sclera_v_min", 150.0),
        sclera_s_max=disk_cfg.get("sclera_s_max", 60.0),
    )
    if disk is None:
        return None

    ring = build_ring_from_disk(
        full_shape,
        disk,
        inner_iris_ratio=disk_cfg.get("inner_iris_ratio", 0.40),
        outer_iris_ratio=disk_cfg.get("outer_iris_ratio", 0.85),
    )
    pupil_method = "iris_disk_estimated" if disk.pupil_estimated else "iris_disk_core"
    return IrisDetectionResult(
        mask=ring.mask,
        center=(disk.center_x, disk.center_y),
        radius=ring.outer_radius,
        eye_side="closeup",
        sample_pixel_count=ring.sample_pixel_count,
        method="eye_closeup_disk",
        pupil_center=(disk.center_x, disk.center_y),
        pupil_radius=disk.pupil_radius,
        inner_radius=ring.inner_radius,
        outer_radius=ring.outer_radius,
        pupil_confidence=disk.confidence,
        iris_confidence=disk.confidence,
        pupil_method=pupil_method,
        iris_outer_method=ring.iris_outer_method,
        candidate_count=0,
        candidate_mask=disk.candidate_mask,
    )


def _precise_acceptable(result: IrisDetectionResult, min_confidence: float) -> bool:
    """精定位是否可信：瞳孔置信度达标且虹膜外缘走了梯度而非倍数兜底。"""
    if result is None or result.sample_pixel_count <= 0:
        return False
    pupil_conf = result.pupil_confidence or 0.0
    return pupil_conf >= min_confidence and result.iris_outer_method == "radial_gradient"


def _detect_from_eye_closeup(
    image_bgr: np.ndarray,
    eye_cfg: dict,
    mode: str = "auto",
) -> Optional[IrisDetectionResult]:
    """
    眼部特写模式：先做镜面反光修复，再两阶段定位虹膜环带。

    mode:
      - auto: 精定位置信度达标用 A，否则回退黑盘直检 B
      - precise: 强制阶段A（清晰精定位）
      - rough: 强制阶段B（实拍黑盘直检）
    不依赖全脸检测。
    """
    specular_cfg = eye_cfg.get("specular", {})
    if specular_cfg.get("enabled", True):
        detect_image, _ = suppress_specular(
            image_bgr,
            v_threshold=specular_cfg.get("v_threshold", 230),
            sat_threshold=specular_cfg.get("sat_threshold", 90),
            max_area_ratio=specular_cfg.get("max_area_ratio", 0.04),
            dilate_px=specular_cfg.get("dilate_px", 3),
        )
    else:
        detect_image = image_bgr

    full_shape = image_bgr.shape

    if mode == "precise":
        return _detect_precise(detect_image, full_shape, eye_cfg)
    if mode == "rough":
        return _detect_rough(detect_image, full_shape, eye_cfg)

    # auto
    min_confidence = eye_cfg.get("auto_precise_min_confidence", 0.5)
    precise = _detect_precise(detect_image, full_shape, eye_cfg)
    if precise is not None and _precise_acceptable(precise, min_confidence):
        return precise

    rough = _detect_rough(detect_image, full_shape, eye_cfg)
    if rough is not None:
        return rough
    return precise


def detect_iris_ring_mask(
    image_bgr: np.ndarray,
    mode: str = "eye_closeup",
    inner_ratio: float = 0.30,
    outer_ratio: float = 0.80,
    eye_closeup_cfg: Optional[dict] = None,
    closeup_mode: str = "auto",
) -> Optional[IrisDetectionResult]:
    """
    检测虹膜环带 mask。

    mode（定位来源）:
      - eye_closeup: 默认，适用于「眼睛占满画面」的拍照/上传图
      - face: 全脸图，MediaPipe 定位
      - auto: 先 eye_closeup，失败再 face

    closeup_mode（眼部特写子策略）:
      - auto: 精定位置信度达标用精定位，否则回退黑盘直检
      - precise: 强制清晰精定位
      - rough: 强制实拍黑盘直检
    """
    eye_cfg = eye_closeup_cfg or {}

    if mode == "eye_closeup":
        return _detect_from_eye_closeup(image_bgr, eye_cfg, mode=closeup_mode)
    if mode == "face":
        return _detect_from_face_landmarks(image_bgr, inner_ratio, outer_ratio)

    # auto
    result = _detect_from_eye_closeup(image_bgr, eye_cfg, mode=closeup_mode)
    if result is not None:
        return result
    return _detect_from_face_landmarks(image_bgr, inner_ratio, outer_ratio)
