"""眼部特写图：基于瞳孔（最暗区域）定位虹膜环带。"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class PupilEstimate:
    """瞳孔估计结果。"""

    center_x: int
    center_y: int
    radius: float


def _detect_pupil_hough(gray: np.ndarray, min_dim: int) -> Optional[PupilEstimate]:
    """Hough 圆检测瞳孔（暗圆），取最接近图像中心的圆。"""
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    max_r = max(min_dim // 3, 10)
    min_r = max(min_dim // 30, 3)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dim // 2,
        param1=80,
        param2=18,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return None

    h, w = gray.shape[:2]
    cx_img, cy_img = w / 2, h / 2
    best = None
    best_score = float("inf")
    for circle in circles[0]:
        cx, cy, r = circle
        dist = (cx - cx_img) ** 2 + (cy - cy_img) ** 2
        score = dist - r * 0.5
        if score < best_score:
            best_score = score
            best = PupilEstimate(center_x=int(cx), center_y=int(cy), radius=float(r))
    return best


def _detect_pupil_dark_region(
    gray: np.ndarray,
    center_roi_ratio: float,
    dark_percentile: float,
) -> Optional[PupilEstimate]:
    """
    在画面中央 ROI 内找最暗连通区域，作为瞳孔。
    适用于「眼睛占满画面」的特写图。
    """
    h, w = gray.shape[:2]
    roi_w = int(w * center_roi_ratio)
    roi_h = int(h * center_roi_ratio)
    x1 = (w - roi_w) // 2
    y1 = (h - roi_h) // 2
    roi = gray[y1 : y1 + roi_h, x1 : x1 + roi_w]

    blurred = cv2.GaussianBlur(roi, (7, 7), 1.5)
    thresh = np.percentile(blurred, dark_percentile)
    dark = (blurred <= thresh).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    roi_cx, roi_cy = roi_w / 2, roi_h / 2
    best = None
    best_score = float("inf")
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        dist = (cx - roi_cx) ** 2 + (cy - roi_cy) ** 2
        (_, _), r = cv2.minEnclosingCircle(cnt)
        score = dist - area * 0.01
        if score < best_score and r >= 2:
            best_score = score
            best = PupilEstimate(
                center_x=int(cx) + x1,
                center_y=int(cy) + y1,
                radius=float(max(r, 3.0)),
            )
    return best


def detect_pupil(
    image_bgr: np.ndarray,
    center_roi_ratio: float = 0.85,
    dark_percentile: float = 12.0,
) -> Optional[PupilEstimate]:
    """综合瞳孔检测：暗区优先，失败则用 Hough 圆。"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    min_dim = min(image_bgr.shape[:2])

    pupil = _detect_pupil_dark_region(gray, center_roi_ratio, dark_percentile)
    if pupil is not None:
        return pupil
    return _detect_pupil_hough(gray, min_dim)


def build_iris_ring_from_pupil(
    image_shape: Tuple[int, int],
    pupil: PupilEstimate,
    inner_pupil_multiplier: float,
    outer_pupil_multiplier: float,
) -> Tuple[np.ndarray, float, float, int]:
    """
    以瞳孔为中心，向外扩展生成虹膜环带 mask。

    inner = pupil_r * inner_pupil_multiplier  （环带内缘，排除瞳孔）
    outer = pupil_r * outer_pupil_multiplier  （环带外缘，虹膜区域）
    """
    h, w = image_shape[:2]
    cx, cy = pupil.center_x, pupil.center_y
    inner_r = pupil.radius * inner_pupil_multiplier
    outer_r = pupil.radius * outer_pupil_multiplier

    max_r = min(w, h) * 0.48
    outer_r = min(outer_r, max_r)
    if outer_r <= inner_r + 2:
        outer_r = inner_r + max(pupil.radius * 0.5, 3)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), int(outer_r), 255, -1)
    cv2.circle(mask, (cx, cy), int(inner_r), 0, -1)
    sample_count = int(np.count_nonzero(mask))
    return mask, outer_r, inner_r, sample_count
