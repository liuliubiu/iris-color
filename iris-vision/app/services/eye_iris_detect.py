"""眼部特写图：基于瞳孔边界与虹膜外缘定位取色环带。"""

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
    confidence: float = 0.0
    method: str = "unknown"
    candidate_count: int = 0
    candidate_mask: Optional[np.ndarray] = None


@dataclass
class IrisRingEstimate:
    """虹膜环带估计结果。"""

    mask: np.ndarray
    outer_radius: float
    inner_radius: float
    sample_pixel_count: int
    iris_outer_method: str
    iris_confidence: float


def _preprocess_luminance(image_bgr: np.ndarray) -> np.ndarray:
    """用亮度通道做轻量归一化，降低局部光照对阈值的影响。"""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(v_channel)
    return cv2.GaussianBlur(enhanced, (5, 5), 1.2)


def _center_roi_bounds(shape: Tuple[int, int], ratio: float) -> Tuple[int, int, int, int]:
    h, w = shape[:2]
    roi_w = int(w * ratio)
    roi_h = int(h * ratio)
    x1 = (w - roi_w) // 2
    y1 = (h - roi_h) // 2
    return x1, y1, roi_w, roi_h


def _detect_pupil_hough(gray: np.ndarray, min_dim: int) -> Optional[PupilEstimate]:
    """Hough 圆检测瞳孔（暗圆），取最接近图像中心的圆。"""
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    max_r = max(int(min_dim * 0.20), 10)
    min_r = max(int(min_dim * 0.025), 3)
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
            best = PupilEstimate(
                center_x=int(cx),
                center_y=int(cy),
                radius=float(r),
                confidence=0.35,
                method="hough_fallback",
            )
    return best


def _detect_pupil_candidates(
    luminance: np.ndarray,
    center_roi_ratio: float,
    dark_percentile: float,
) -> Optional[PupilEstimate]:
    """
    在画面中央 ROI 内找极暗、近圆且半径合理的连通域。
    深棕虹膜也偏暗，所以这里会主动拒绝半径过大的暗区。
    """
    h, w = luminance.shape[:2]
    min_dim = min(h, w)
    x1, y1, roi_w, roi_h = _center_roi_bounds(luminance.shape, center_roi_ratio)
    roi = luminance[y1 : y1 + roi_h, x1 : x1 + roi_w]

    otsu_threshold, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strict_percentile = min(dark_percentile, 4.0)
    percentile_threshold = float(np.percentile(roi, strict_percentile))
    thresh = min(percentile_threshold, float(otsu_threshold) * 0.70)
    dark = (roi <= thresh).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        full_mask = np.zeros_like(luminance, dtype=np.uint8)
        full_mask[y1 : y1 + roi_h, x1 : x1 + roi_w] = dark
        return PupilEstimate(0, 0, 0, candidate_count=0, candidate_mask=full_mask)

    roi_cx, roi_cy = roi_w / 2, roi_h / 2
    best = None
    best_score = -float("inf")
    candidate_count = 0
    min_radius = max(min_dim * 0.025, 3.0)
    max_radius = max(min_dim * 0.18, min_radius + 1.0)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < np.pi * min_radius * min_radius * 0.45:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        (_, _), enclosing_r = cv2.minEnclosingCircle(cnt)
        equiv_r = (area / np.pi) ** 0.5
        radius = float((enclosing_r + equiv_r) / 2.0)
        if radius < min_radius or radius > max_radius:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        if circularity < 0.45:
            continue

        candidate_count += 1
        dist_norm = ((cx - roi_cx) ** 2 + (cy - roi_cy) ** 2) ** 0.5 / max(roi_w, roi_h)
        fill_ratio = area / max(np.pi * enclosing_r * enclosing_r, 1.0)
        mean_lum = float(cv2.mean(roi, mask=cv2.drawContours(np.zeros_like(roi), [cnt], -1, 255, -1))[0])
        darkness = 1.0 - min(mean_lum / 255.0, 1.0)
        radius_penalty = abs(radius - min_dim * 0.10) / max(min_dim * 0.10, 1.0)
        score = (
            darkness * 2.0
            + circularity * 1.2
            + fill_ratio
            - dist_norm * 1.8
            - radius_penalty * 0.35
        )
        if score > best_score:
            best_score = score
            best = PupilEstimate(
                center_x=int(cx) + x1,
                center_y=int(cy) + y1,
                radius=float(max(radius, 3.0)),
                confidence=float(np.clip((score + 0.5) / 4.0, 0.1, 0.85)),
                method="dark_candidate",
            )

    full_mask = np.zeros_like(luminance, dtype=np.uint8)
    full_mask[y1 : y1 + roi_h, x1 : x1 + roi_w] = dark
    if best is not None:
        best.candidate_count = candidate_count
        best.candidate_mask = full_mask
        return best
    return PupilEstimate(0, 0, 0, candidate_count=0, candidate_mask=full_mask)


def _sample_luminance_at(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x_clipped = np.clip(np.rint(x).astype(np.int32), 0, w - 1)
    y_clipped = np.clip(np.rint(y).astype(np.int32), 0, h - 1)
    return image[y_clipped, x_clipped].astype(np.float32)


def _smooth_profile(values: np.ndarray, window: int = 7) -> np.ndarray:
    """平滑一维径向亮度，避免卷积边缘产生假梯度。"""
    if len(values) < window:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _refine_pupil_boundary(luminance: np.ndarray, pupil: PupilEstimate) -> PupilEstimate:
    """沿射线寻找瞳孔黑区到虹膜的亮度上升边界。"""
    h, w = luminance.shape[:2]
    min_dim = min(h, w)
    cx, cy = float(pupil.center_x), float(pupil.center_y)
    if pupil.radius <= 0:
        return pupil

    radii = []
    angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    start_r = max(pupil.radius * 0.55, min_dim * 0.015)
    end_r = min(max(pupil.radius * 1.75, min_dim * 0.16), min_dim * 0.26)
    radial_samples = np.linspace(start_r, end_r, 90)

    for angle in angles:
        xs = cx + np.cos(angle) * radial_samples
        ys = cy + np.sin(angle) * radial_samples
        values = _sample_luminance_at(luminance, xs, ys)
        gradients = np.diff(values)
        if len(gradients) == 0:
            continue
        idx = int(np.argmax(gradients))
        if gradients[idx] < 3.0:
            continue
        radii.append(float(radial_samples[idx]))

    if len(radii) < 18:
        pupil.method = f"{pupil.method}+unrefined"
        return pupil

    radii_arr = np.asarray(radii, dtype=np.float32)
    median = float(np.median(radii_arr))
    mad = float(np.median(np.abs(radii_arr - median))) or 1.0
    inliers = radii_arr[np.abs(radii_arr - median) <= 2.5 * mad]
    if len(inliers) < 12:
        return pupil

    refined_radius = float(np.median(inliers))
    if refined_radius < min_dim * 0.02 or refined_radius > min_dim * 0.22:
        return pupil

    support = len(inliers) / len(angles)
    confidence = max(pupil.confidence, float(np.clip(0.45 + support * 0.55, 0.45, 0.98)))
    return PupilEstimate(
        center_x=pupil.center_x,
        center_y=pupil.center_y,
        radius=refined_radius,
        confidence=confidence,
        method=f"{pupil.method}+radial_refine",
        candidate_count=pupil.candidate_count,
        candidate_mask=pupil.candidate_mask,
    )


def _estimate_iris_outer_radius(
    luminance: np.ndarray,
    pupil: PupilEstimate,
    outer_pupil_multiplier: float,
) -> Tuple[float, str, float]:
    """沿多条射线寻找虹膜到巩膜/皮肤的外缘。"""
    h, w = luminance.shape[:2]
    min_dim = min(h, w)
    cx, cy = float(pupil.center_x), float(pupil.center_y)
    start_r = max(pupil.radius * 1.65, min_dim * 0.10)
    end_r = min(min_dim * 0.48, max(pupil.radius * 4.2, start_r + min_dim * 0.12))

    if end_r <= start_r + 4:
        fallback = min(pupil.radius * outer_pupil_multiplier, min_dim * 0.48)
        return fallback, "fallback_multiplier", 0.25

    radii = []
    angles = np.deg2rad(
        [-45, -35, -25, -15, 0, 15, 25, 35, 45, 135, 145, 155, 165, 180, 195, 205, 215, 225]
    )
    radial_samples = np.linspace(start_r, end_r, 120)
    for angle in angles:
        xs = cx + np.cos(angle) * radial_samples
        ys = cy + np.sin(angle) * radial_samples
        values = _sample_luminance_at(luminance, xs, ys)
        smooth = _smooth_profile(values, 7)
        gradients = np.diff(smooth)
        if len(gradients) == 0:
            continue
        idx = int(np.argmax(gradients))
        if idx < 4 or idx > len(gradients) - 5:
            continue
        local_gain = float(gradients[idx])
        total_gain = float(smooth[min(idx + 5, len(smooth) - 1)] - smooth[max(idx - 5, 0)])
        if local_gain < 2.0 and total_gain < 7.0:
            continue
        radii.append(float(radial_samples[idx]))

    if len(radii) < 5:
        fallback = min(pupil.radius * outer_pupil_multiplier, min_dim * 0.48)
        return fallback, "fallback_multiplier", 0.25

    radii_arr = np.asarray(radii, dtype=np.float32)
    median = float(np.median(radii_arr))
    mad = float(np.median(np.abs(radii_arr - median))) or 1.0
    inliers = radii_arr[np.abs(radii_arr - median) <= 2.5 * mad]
    if len(inliers) < 4:
        fallback = min(pupil.radius * outer_pupil_multiplier, min_dim * 0.48)
        return fallback, "fallback_multiplier", 0.25

    outer_radius = float(np.median(inliers))
    confidence = float(np.clip(len(inliers) / len(angles), 0.3, 0.95))
    return outer_radius, "radial_gradient", confidence


def detect_pupil(
    image_bgr: np.ndarray,
    center_roi_ratio: float = 0.85,
    dark_percentile: float = 12.0,
) -> Optional[PupilEstimate]:
    """综合瞳孔检测：严格暗区候选优先，失败则用 Hough 圆。"""
    luminance = _preprocess_luminance(image_bgr)
    min_dim = min(image_bgr.shape[:2])

    pupil = _detect_pupil_candidates(luminance, center_roi_ratio, dark_percentile)
    if pupil is not None and pupil.radius > 0:
        return _refine_pupil_boundary(luminance, pupil)

    fallback = _detect_pupil_hough(luminance, min_dim)
    if fallback is not None and pupil is not None:
        fallback.candidate_mask = pupil.candidate_mask
    return fallback


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


def build_refined_iris_ring(
    image_bgr: np.ndarray,
    pupil: PupilEstimate,
    inner_pupil_multiplier: float,
    outer_pupil_multiplier: float,
    inner_iris_ratio: float = 0.35,
    outer_iris_ratio: float = 0.85,
) -> IrisRingEstimate:
    """
    使用独立估计的虹膜外缘生成取色环带。

    环带以内外边距约束在虹膜纹理中段，降低瞳孔边缘、巩膜和眼睑混入。
    """
    h, w = image_bgr.shape[:2]
    min_dim = min(h, w)
    luminance = _preprocess_luminance(image_bgr)
    iris_outer_r, iris_outer_method, iris_confidence = _estimate_iris_outer_radius(
        luminance,
        pupil,
        outer_pupil_multiplier,
    )
    iris_outer_r = float(np.clip(iris_outer_r, pupil.radius * 1.8, min_dim * 0.48))

    pupil_margin = max(pupil.radius * (inner_pupil_multiplier - 1.0), 2.0)
    inner_r = max(pupil.radius + pupil_margin, iris_outer_r * inner_iris_ratio)
    outer_r = min(iris_outer_r * outer_iris_ratio, iris_outer_r - max(min_dim * 0.01, 2.0))
    if outer_r <= inner_r + 3:
        inner_r = min(pupil.radius * inner_pupil_multiplier, iris_outer_r - 6)
        outer_r = iris_outer_r - 3

    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = pupil.center_x, pupil.center_y
    cv2.circle(mask, (cx, cy), int(max(outer_r, 1)), 255, -1)
    cv2.circle(mask, (cx, cy), int(max(inner_r, 1)), 0, -1)
    sample_count = int(np.count_nonzero(mask))
    return IrisRingEstimate(
        mask=mask,
        outer_radius=float(outer_r),
        inner_radius=float(inner_r),
        sample_pixel_count=sample_count,
        iris_outer_method=iris_outer_method,
        iris_confidence=iris_confidence,
    )
