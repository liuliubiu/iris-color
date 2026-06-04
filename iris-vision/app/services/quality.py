"""图像质量检测：模糊、过曝、睁眼（模式可配置）。"""

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from app.services.face_landmarker import detect_face_landmarks

_LEFT_EYE_TOP = 159
_LEFT_EYE_BOTTOM = 145
_RIGHT_EYE_TOP = 386
_RIGHT_EYE_BOTTOM = 374


@dataclass
class QualityResult:
    """质量检测结果。"""

    blur_score: float
    overexposed_ratio: float
    eye_open: bool
    issues: List[str]
    passed: bool


def compute_blur_score(image_bgr: np.ndarray) -> float:
    """拉普拉斯方差，越大越清晰。"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_overexposed_ratio(image_bgr: np.ndarray, threshold: int = 250) -> float:
    """RGB 三通道均高于 threshold 的像素占比。"""
    over = np.all(image_bgr >= threshold, axis=2)
    return float(np.mean(over))


def _is_eye_open_via_face(image_bgr: np.ndarray) -> bool:
    """全脸模式：用眼睑 landmark 判断睁眼。"""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    try:
        landmarks = detect_face_landmarks(rgb)
    except (FileNotFoundError, RuntimeError, ValueError):
        return True
    if landmarks is None:
        return True

    h = image_bgr.shape[0]

    def eye_open_distance(top_idx: int, bottom_idx: int) -> float:
        top = landmarks[top_idx]
        bottom = landmarks[bottom_idx]
        return abs(top.y - bottom.y) * h

    left_open = eye_open_distance(_LEFT_EYE_TOP, _LEFT_EYE_BOTTOM)
    right_open = eye_open_distance(_RIGHT_EYE_TOP, _RIGHT_EYE_BOTTOM)
    return (left_open + right_open) / 2.0 > 3.0


def check_quality(
    image_bgr: np.ndarray,
    blur_threshold: float,
    overexposed_ratio_max: float,
    detection_mode: str = "eye_closeup",
) -> QualityResult:
    """
    综合质量检测。

    eye_closeup 模式不做全脸「闭眼」检测（特写图通常无法检出人脸 landmark）。
    """
    issues: List[str] = []
    blur_score = compute_blur_score(image_bgr)
    overexposed_ratio = compute_overexposed_ratio(image_bgr)

    if detection_mode in ("face", "auto"):
        eye_open = _is_eye_open_via_face(image_bgr)
    else:
        # 眼部特写：默认视为睁眼，后续由瞳孔/虹膜是否检出判断
        eye_open = True

    if blur_score < blur_threshold:
        issues.append("image_too_blurry")
    if overexposed_ratio > overexposed_ratio_max:
        issues.append("image_overexposed")
    if not eye_open:
        issues.append("eye_closed")

    return QualityResult(
        blur_score=blur_score,
        overexposed_ratio=overexposed_ratio,
        eye_open=eye_open,
        issues=issues,
        passed=len(issues) == 0,
    )
