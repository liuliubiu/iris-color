"""图像质量检测：模糊、过曝、睁眼。"""

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from app.services.face_landmarker import detect_face_landmarks

# 左右眼上下眼睑 landmark 索引（MediaPipe Face Mesh）
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
    """
    用拉普拉斯方差衡量清晰度。
    值越大越清晰；低于阈值则认为模糊。
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_overexposed_ratio(image_bgr: np.ndarray, threshold: int = 250) -> float:
    """计算过曝像素占比（RGB 三通道均高于阈值）。"""
    over = np.all(image_bgr >= threshold, axis=2)
    return float(np.mean(over))


def is_eye_open(image_bgr: np.ndarray) -> bool:
    """
    通过眼睑 landmark 垂直距离判断睁眼。
    若未检测到人脸，默认 True（避免误拒）。
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    landmarks = detect_face_landmarks(rgb)
    if landmarks is None:
        return True

    h, w = image_bgr.shape[:2]

    def eye_open_distance(top_idx: int, bottom_idx: int) -> float:
        top = landmarks[top_idx]
        bottom = landmarks[bottom_idx]
        return abs(top.y - bottom.y) * h

    left_open = eye_open_distance(_LEFT_EYE_TOP, _LEFT_EYE_BOTTOM)
    right_open = eye_open_distance(_RIGHT_EYE_TOP, _RIGHT_EYE_BOTTOM)
    avg_open = (left_open + right_open) / 2.0
    return avg_open > 3.0


def check_quality(
    image_bgr: np.ndarray,
    blur_threshold: float,
    overexposed_ratio_max: float,
) -> QualityResult:
    """
    综合质量检测。
    返回 issues 列表；passed=False 时上层应拒绝分析。
    """
    issues: List[str] = []
    blur_score = compute_blur_score(image_bgr)
    overexposed_ratio = compute_overexposed_ratio(image_bgr)
    eye_open = is_eye_open(image_bgr)

    if blur_score < blur_threshold:
        issues.append("image_too_blurry")
    if overexposed_ratio > overexposed_ratio_max:
        issues.append("image_overexposed")
    if not eye_open:
        issues.append("eye_closed")

    passed = len(issues) == 0
    return QualityResult(
        blur_score=blur_score,
        overexposed_ratio=overexposed_ratio,
        eye_open=eye_open,
        issues=issues,
        passed=passed,
    )
