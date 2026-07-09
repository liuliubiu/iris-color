"""图像质量检测：模糊、过曝（眼部特写）。"""

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from app.services.scope_field import ScopeField


QUALITY_ISSUE_MESSAGES: dict[str, str] = {
    "image_too_blurry": "图像模糊，请重新对焦拍摄",
    "image_overexposed": "图像过曝，请避免强光直射",
}


def format_quality_failure_message(issues: List[str]) -> str:
    """将质量问题代码转为用户可读提示。"""
    parts = [QUALITY_ISSUE_MESSAGES.get(code, code) for code in issues]
    if not parts:
        return "图像质量未达标，请重新拍摄更清晰的眼部特写"
    return "图像质量未达标：" + "；".join(parts)


@dataclass
class QualityResult:
    """质量检测结果。"""

    blur_score: float
    overexposed_ratio: float
    eye_open: bool
    issues: List[str]
    passed: bool


def _scope_interior_mask(shape, scope: Optional[ScopeField]) -> Optional[np.ndarray]:
    """视场圆内部 bool mask；无视场时返回 None（全图统计）。"""
    if scope is None:
        return None
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(
        mask,
        (int(round(scope.center_x)), int(round(scope.center_y))),
        max(int(scope.radius * 0.98), 1),
        255,
        -1,
    )
    return mask > 0


def compute_blur_score(image_bgr: np.ndarray, scope: Optional[ScopeField] = None) -> float:
    """拉普拉斯方差，越大越清晰。镜筒图只统计视场圆内（黑边方差≈0 会稀释分数）。"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    interior = _scope_interior_mask(gray.shape, scope)
    if interior is not None and np.any(interior):
        return float(lap[interior].var())
    return float(lap.var())


def compute_overexposed_ratio(
    image_bgr: np.ndarray, threshold: int = 250, scope: Optional[ScopeField] = None
) -> float:
    """RGB 三通道均高于 threshold 的像素占比。镜筒图只统计视场圆内。"""
    over = np.all(image_bgr >= threshold, axis=2)
    interior = _scope_interior_mask(over.shape, scope)
    if interior is not None and np.any(interior):
        return float(np.mean(over[interior]))
    return float(np.mean(over))


def check_quality(
    image_bgr: np.ndarray,
    blur_threshold: float,
    overexposed_ratio_max: float,
    scope: Optional[ScopeField] = None,
) -> QualityResult:
    """
    综合质量检测（眼部特写）。

    眼部特写不做「闭眼」检测（特写图无法检出人脸 landmark），
    由后续瞳孔/虹膜是否检出间接判断。scope 非空（镜筒特写）时
    模糊/过曝只统计视场圆内。
    """
    issues: List[str] = []
    blur_score = compute_blur_score(image_bgr, scope=scope)
    overexposed_ratio = compute_overexposed_ratio(image_bgr, scope=scope)

    if blur_score < blur_threshold:
        issues.append("image_too_blurry")
    if overexposed_ratio > overexposed_ratio_max:
        issues.append("image_overexposed")

    return QualityResult(
        blur_score=blur_score,
        overexposed_ratio=overexposed_ratio,
        eye_open=True,
        issues=issues,
        passed=len(issues) == 0,
    )
