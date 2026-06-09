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


@dataclass
class IrisDiskEstimate:
    """黑盘直检结果：把瞳孔+虹膜当作一个暗圆盘定位。"""

    center_x: int
    center_y: int
    iris_radius: float
    pupil_radius: float
    confidence: float
    method: str = "iris_disk"
    pupil_estimated: bool = True
    candidate_mask: Optional[np.ndarray] = None


def suppress_specular(
    image_bgr: np.ndarray,
    v_threshold: int = 230,
    sat_threshold: int = 90,
    max_area_ratio: float = 0.04,
    dilate_px: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    检测镜面反光小亮斑并 inpaint 填充。

    仅修复「高亮度 + 低饱和 + 小面积」的反光点（瞳孔亮环、屏幕/窗户反光），
    不会动巩膜、皮肤这种大片高亮区域。返回 (修复后 BGR, 反光 mask)。
    """
    h, w = image_bgr.shape[:2]
    empty = np.zeros((h, w), dtype=np.uint8)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]

    bright = ((v_channel >= v_threshold) & (s_channel <= sat_threshold)).astype(np.uint8) * 255
    if not np.any(bright):
        return image_bgr, empty

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    spec_mask = np.zeros((h, w), dtype=np.uint8)
    max_area = max(max_area_ratio * h * w, 1.0)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] <= max_area:
            spec_mask[labels == i] = 255

    if not np.any(spec_mask):
        return image_bgr, spec_mask

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        spec_mask = cv2.dilate(spec_mask, kernel)

    inpainted = cv2.inpaint(image_bgr, spec_mask, 4, cv2.INPAINT_TELEA)
    return inpainted, spec_mask


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


def _ray_spec_fraction(
    spec_mask: Optional[np.ndarray], xs: np.ndarray, ys: np.ndarray
) -> float:
    """一条射线上落在镜面反光区域的采样点占比（用于跳过被反光污染的射线）。"""
    if spec_mask is None:
        return 0.0
    h, w = spec_mask.shape[:2]
    xi = np.clip(np.rint(xs).astype(np.int32), 0, w - 1)
    yi = np.clip(np.rint(ys).astype(np.int32), 0, h - 1)
    return float(np.mean(spec_mask[yi, xi] > 0))


def _smooth_profile(values: np.ndarray, window: int = 7) -> np.ndarray:
    """平滑一维径向亮度，避免卷积边缘产生假梯度。"""
    if len(values) < window:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _refine_pupil_boundary(
    luminance: np.ndarray,
    pupil: PupilEstimate,
    spec_mask: Optional[np.ndarray] = None,
) -> PupilEstimate:
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
        # 反光会在瞳孔亮环处制造假的亮度跃升，污染严重的射线直接跳过
        if _ray_spec_fraction(spec_mask, xs, ys) > 0.3:
            continue
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
    spec_mask: Optional[np.ndarray] = None,
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
        if _ray_spec_fraction(spec_mask, xs, ys) > 0.3:
            continue
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
    spec_mask: Optional[np.ndarray] = None,
) -> Optional[PupilEstimate]:
    """综合瞳孔检测：严格暗区候选优先，失败则用 Hough 圆。"""
    luminance = _preprocess_luminance(image_bgr)
    min_dim = min(image_bgr.shape[:2])

    pupil = _detect_pupil_candidates(luminance, center_roi_ratio, dark_percentile)
    if pupil is not None and pupil.radius > 0:
        return _refine_pupil_boundary(luminance, pupil, spec_mask)

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
    spec_mask: Optional[np.ndarray] = None,
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
        spec_mask,
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


def _surround_support(
    luminance: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    inside_mean: float,
    margin: float,
) -> float:
    """采样外侧环带，统计「外亮于内」满足的角度比例（巩膜/皮肤包围验证）。"""
    support, _ = _surround_stats(luminance, cx, cy, radius, inside_mean, margin)
    return support


def _surround_stats(
    luminance: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    inside_mean: float,
    margin: float,
) -> Tuple[float, float]:
    """返回 (外亮于内的角度占比, 外侧平均亮度)。"""
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    out_r = radius * 1.18
    xs = cx + np.cos(angles) * out_r
    ys = cy + np.sin(angles) * out_r
    values = _sample_luminance_at(luminance, xs, ys)
    support = float(np.mean(values > inside_mean + margin))
    return support, float(np.mean(values))


def _estimate_pupil_in_disk(
    luminance: np.ndarray,
    cx: float,
    cy: float,
    iris_r: float,
    pupil_iris_ratio: float,
) -> Tuple[float, bool]:
    """
    在黑盘内尝试找更暗的瞳孔核。

    深色融合眼里瞳孔与虹膜亮度几乎一致，找不到时按 pupil_iris_ratio 估算。
    返回 (pupil_radius, estimated)。estimated=True 表示用比例兜底。
    """
    h, w = luminance.shape[:2]
    fallback = float(iris_r * pupil_iris_ratio)
    r_search = int(iris_r * 0.75)
    x1 = max(int(cx - r_search), 0)
    x2 = min(int(cx + r_search), w)
    y1 = max(int(cy - r_search), 0)
    y2 = min(int(cy + r_search), h)
    roi = luminance[y1:y2, x1:x2]
    if roi.size == 0:
        return fallback, True

    thr = float(np.percentile(roi, 15))
    dark = (roi <= thr).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    local_cx, local_cy = cx - x1, cy - y1
    min_r = iris_r * 0.12
    max_r = iris_r * 0.65
    best_r = None
    best_dist = float("inf")
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < np.pi * min_r * min_r:
            continue
        (ex, ey), enclosing_r = cv2.minEnclosingCircle(cnt)
        if enclosing_r < min_r or enclosing_r > max_r:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.5:
            continue
        dist = (ex - local_cx) ** 2 + (ey - local_cy) ** 2
        if dist < best_dist:
            best_dist = dist
            best_r = float(enclosing_r)

    if best_r is not None:
        return best_r, False
    return fallback, True


def _inside_mean(luminance: np.ndarray, cx: float, cy: float, radius: float) -> float:
    """圆盘内（取 0.7r 避开边界）平均亮度。"""
    mask = np.zeros(luminance.shape, dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), max(int(radius * 0.7), 1), 255, -1)
    return float(cv2.mean(luminance, mask=mask)[0])


def _sclera_fraction(
    hsv: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    v_min: float,
    s_max: float,
) -> float:
    """外侧环带中「偏白巩膜」（高亮度、低饱和）像素占比，用于区分虹膜与眉毛。"""
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    out_r = radius * 1.15
    xs = cx + np.cos(angles) * out_r
    ys = cy + np.sin(angles) * out_r
    s_vals = _sample_luminance_at(hsv[:, :, 1], xs, ys)
    v_vals = _sample_luminance_at(hsv[:, :, 2], xs, ys)
    whitish = (v_vals > v_min) & (s_vals < s_max)
    return float(np.mean(whitish))


def _bilateral_sclera_fraction(
    hsv: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    v_min: float,
    s_max: float,
) -> float:
    """
    左右两侧楔形各自的「偏白巩膜」占比，取较小值。

    真实虹膜被巩膜水平包围，左右两侧都应有白色；眉毛/睫毛被皮肤包围，
    两侧都没有白巩膜。取双侧最小值可拒绝「单侧偶然有亮皮肤」的假阳性。
    """
    out_r = radius * 1.15

    def _wedge(center_deg: float) -> float:
        angles = np.deg2rad(np.linspace(center_deg - 35.0, center_deg + 35.0, 13))
        xs = cx + np.cos(angles) * out_r
        ys = cy + np.sin(angles) * out_r
        s_vals = _sample_luminance_at(hsv[:, :, 1], xs, ys)
        v_vals = _sample_luminance_at(hsv[:, :, 2], xs, ys)
        return float(np.mean((v_vals > v_min) & (s_vals < s_max)))

    right = _wedge(0.0)
    left = _wedge(180.0)
    return min(left, right)


def _local_edge_support(
    luminance: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    min_step: float,
) -> float:
    """边界处「紧邻外侧亮于紧邻内侧」的角度占比（局部亮度跃升）。

    真实虹膜圆的边界几乎处处是「虹膜→巩膜/眼睑」的真实亮度边；而锁在眉毛上的
    大圈，其顶部/两侧边界穿过的是「皮肤→皮肤」，没有跃升。与 _surround_stats
    （对比全局内部均值）不同，这里比同一角度的内外局部，能拒绝「圈内混入大片亮肤」
    的眉毛误检（眉毛误检常因全局对比/巩膜占比虚高而被选中）。
    """
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    in_r = radius * 0.85
    out_r = radius * 1.15
    v_in = _sample_luminance_at(
        luminance, cx + np.cos(angles) * in_r, cy + np.sin(angles) * in_r
    )
    v_out = _sample_luminance_at(
        luminance, cx + np.cos(angles) * out_r, cy + np.sin(angles) * out_r
    )
    return float(np.mean((v_out - v_in) > min_step))


def _dark_core_compactness(
    luminance: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
) -> float:
    """候选圆内暗核的「居中且致密」程度，[0,1]。

    虹膜/瞳孔暗核居中、近圆；眉毛暗带偏上、细长。取圆内暗像素的最大连通域，
    综合其质心居中度、近圆度与填充度。仅作加分项（与巩膜证据正交）。
    """
    h, w = luminance.shape[:2]
    r = int(max(radius, 2))
    x1, y1 = max(int(cx - r), 0), max(int(cy - r), 0)
    x2, y2 = min(int(cx + r), w), min(int(cy + r), h)
    roi = luminance[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    local_cx, local_cy = cx - x1, cy - y1
    circ = np.zeros(roi.shape, dtype=np.uint8)
    cv2.circle(circ, (int(local_cx), int(local_cy)), r, 255, -1)
    inside = roi[circ > 0]
    if inside.size == 0:
        return 0.0
    thr = float(np.percentile(inside, 45))
    dark = ((roi <= thr) & (circ > 0)).astype(np.uint8)
    num, _, stats, centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if num <= 1:
        return 0.0
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = float(stats[largest, cv2.CC_STAT_AREA])
    ccx, ccy = centroids[largest]
    offset = ((ccx - local_cx) ** 2 + (ccy - local_cy) ** 2) ** 0.5 / max(r, 1)
    bw = float(stats[largest, cv2.CC_STAT_WIDTH])
    bh = float(stats[largest, cv2.CC_STAT_HEIGHT])
    aspect = max(bw, bh) / max(min(bw, bh), 1.0)
    fill = area / max(np.pi * r * r, 1.0)
    centered = max(0.0, 1.0 - offset)
    roundness = max(0.0, 1.0 - (aspect - 1.0) / 2.0)
    return float(np.clip(centered * 0.5 + roundness * 0.3 + min(fill / 0.4, 1.0) * 0.2, 0.0, 1.0))


def _circle_edge_strength(
    luminance: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    cos_a: np.ndarray,
    sin_a: np.ndarray,
) -> float:
    """圆周「紧邻外侧 - 紧邻内侧」平均亮度（Daugman 角向积分的径向导数近似）。

    真实虹膜/瞳孔边界处处是亮度跃变；锁在眉毛/皮肤上的大圈只有局部弧段有边，均值低。
    实测此量能把 1026/IMG_* 的真实眼/虹膜在全部候选里排到第 1（而旧的硬对比门槛会
    误杀「深虹膜被深眼睑包围」这种低对比真虹膜）。
    """
    v_in = _sample_luminance_at(luminance, cx + cos_a * r * 0.88, cy + sin_a * r * 0.88)
    v_out = _sample_luminance_at(luminance, cx + cos_a * r * 1.14, cy + sin_a * r * 1.14)
    return float(np.mean(v_out - v_in))


def _refine_center_edge(
    luminance: np.ndarray,
    cx: float,
    cy: float,
    r0: float,
    cos_a: np.ndarray,
    sin_a: np.ndarray,
) -> Tuple[float, float]:
    """种子中心邻域内微调，最大化多半径圆周边界强度之和（锁定同心圆心）。"""
    h, w = luminance.shape[:2]
    span = max(r0 * 0.3, 6.0)
    step = max(span / 6.0, 1.5)
    radii = (r0 * 0.7, r0, r0 * 1.3)
    best = (float(cx), float(cy))
    best_s = -1e9
    for ddx in np.arange(-span, span + 1, step):
        for ddy in np.arange(-span, span + 1, step):
            ncx, ncy = cx + ddx, cy + ddy
            if ncx < 0 or ncx >= w or ncy < 0 or ncy >= h:
                continue
            s = sum(_circle_edge_strength(luminance, ncx, ncy, r, cos_a, sin_a) for r in radii)
            if s > best_s:
                best_s = s
                best = (float(ncx), float(ncy))
    return best


def _find_profile_peaks(rs: np.ndarray, prof: np.ndarray, min_strength: float) -> list:
    """径向边界强度剖面上的局部极大（峰），返回 [(r, strength), ...] 按 r 升序。"""
    peaks = []
    for i in range(1, len(prof) - 1):
        if prof[i] >= prof[i - 1] and prof[i] > prof[i + 1] and prof[i] >= min_strength:
            peaks.append((float(rs[i]), float(prof[i])))
    return peaks


def _select_iris_by_edge(
    luminance: np.ndarray,
    candidates,
    sw: int,
    sh: int,
    min_r: int,
    max_r: int,
    center_bias: float = 6.0,
    peak_min_strength: float = 7.0,
) -> Optional[Tuple[float, float, float, Optional[float], float]]:
    """边界强度选中心 + 径向剖面分离瞳孔/虹膜缘（方案 A 核心）。

    返回 (cx, cy, iris_r, pupil_r_or_None, edge_strength_at_iris)，坐标在降采样空间。
    1) 选圆周边界最强的候选作中心种子（无硬门槛，仅轻微居中偏好，避免锁到边角强纹理）；
    2) 邻域微调中心；
    3) 沿半径算边界强度剖面：最强峰若另有「明显更大且够强」的外峰，则该外峰=虹膜缘、强峰=瞳孔；
       否则强峰本身=虹膜缘，更内的显著峰=瞳孔。
    """
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    img_cx, img_cy = sw / 2.0, sh / 2.0
    diag = float(max(sw, sh))

    seed = None
    seed_score = -1e9
    for cx, cy, r in candidates:
        s = _circle_edge_strength(luminance, cx, cy, r, cos_a, sin_a)
        dist = ((cx - img_cx) ** 2 + (cy - img_cy) ** 2) ** 0.5 / diag
        s_adj = s - dist * center_bias
        if s_adj > seed_score:
            seed_score = s_adj
            seed = (cx, cy, r)
    if seed is None:
        return None

    cx, cy, r0 = seed
    cx, cy = _refine_center_edge(luminance, cx, cy, r0, cos_a, sin_a)

    rs = np.arange(max(min_r * 0.6, 6.0), float(max_r), max(1.0, (max_r - min_r) / 140.0))
    prof = _smooth_profile(
        np.array([_circle_edge_strength(luminance, cx, cy, r, cos_a, sin_a) for r in rs]), 5
    )
    peaks = _find_profile_peaks(rs, prof, peak_min_strength)
    if not peaks:
        iris_r = r0
        pupil_r = None
    else:
        strongest = max(peaks, key=lambda p: p[1])
        outer = [p for p in peaks if p[0] >= strongest[0] * 1.4 and p[1] >= max(10.0, strongest[1] * 0.30)]
        if outer:
            iris_r = max(outer, key=lambda p: p[1])[0]
            pupil_r = strongest[0]
        else:
            iris_r = strongest[0]
            inner = [p for p in peaks if p[0] <= iris_r * 0.65 and p[1] >= 10.0]
            pupil_r = min(inner, key=lambda p: p[0])[0] if inner else None

    edge_iris = _circle_edge_strength(luminance, cx, cy, iris_r, cos_a, sin_a)
    return float(cx), float(cy), float(iris_r), pupil_r, edge_iris


def detect_iris_disk(
    image_bgr: np.ndarray,
    dark_percentile: float = 35.0,
    min_radius_ratio: float = 0.12,
    max_radius_ratio: float = 0.45,
    circularity_min: float = 0.55,
    surround_contrast: float = 8.0,
    surround_support_min: float = 0.45,
    pupil_iris_ratio: float = 0.30,
    sclera_min: float = 0.10,
    sclera_v_min: float = 150.0,
    sclera_s_max: float = 60.0,
    sclera_bilateral_min: float = 0.0,
    target_min_dim: int = 700,
    sclera_global_weight: float = 0.8,
    bilateral_weight: float = 0.8,
    local_edge_weight: float = 0.0,
    local_edge_min_step: float = 12.0,
    compactness_weight: float = 0.0,
    use_edge_ranking: bool = True,
    edge_min_radius_ratio: float = 0.07,
    edge_center_bias: float = 6.0,
    edge_hough_param2: int = 16,
) -> Optional[IrisDiskEstimate]:
    """
    黑盘直检：直接用 Hough 找虹膜/巩膜的圆形边界，再用「内暗外亮」验证。

    适用于瞳孔与虹膜融成一团黑、且与睫毛/眉毛/眼窝阴影连成一片的实拍图——
    此时暗区连通域不再是圆，必须靠圆形边界（梯度）而非暗区面积定位。
    """
    h, w = image_bgr.shape[:2]
    min_dim = min(h, w)

    # 大图先降采样，兼顾 Hough 速度与稳定性
    target = target_min_dim
    scale = target / min_dim if min_dim > target else 1.0
    small = (
        cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else image_bgr
    )
    luminance = _preprocess_luminance(small)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    sh, sw = luminance.shape[:2]
    s_min_dim = min(sh, sw)

    min_r = max(int(s_min_dim * min_radius_ratio), 8)
    max_r = max(int(s_min_dim * max_radius_ratio), min_r + 1)
    blurred = cv2.medianBlur(luminance, 5)

    # 候选半径下限：边界排序模式放宽（让偏小的真实虹膜/瞳孔也进候选），旧模式用 min_r
    hough_min_r = max(int(s_min_dim * edge_min_radius_ratio), 6) if use_edge_ranking else min_r

    # 旧打分模式多档短路；边界排序模式只跑单档（HoughCircles 每次约 2s，多档会拖到 8-9s）。
    # 单档用较低 param2 一次拿到强+弱候选（含 1026 这种低对比真虹膜），由边界强度再排序。
    param2_levels = (edge_hough_param2,) if use_edge_ranking else (30, 22, 16)
    candidates: dict = {}
    for param2 in param2_levels:
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=hough_min_r,
            param1=50,
            param2=param2,
            minRadius=hough_min_r,
            maxRadius=max_r,
        )
        if circles is not None:
            for circle in circles[0]:
                cx, cy, radius = float(circle[0]), float(circle[1]), float(circle[2])
                key = (round(cx / 12), round(cy / 12), round(radius / 12))
                candidates.setdefault(key, (cx, cy, radius))
        if not use_edge_ranking and len(candidates) >= 2:
            break
    if not candidates:
        return None

    # 方案 A：Daugman 式圆周边界强度选中心 + 径向剖面定虹膜缘（解决眉毛/皮肤脱靶误检）
    if use_edge_ranking:
        picked = _select_iris_by_edge(
            luminance, list(candidates.values()), sw, sh, min_r, max_r,
            center_bias=edge_center_bias,
        )
        if picked is None:
            return None
        cx, cy, iris_r, pupil_seed, edge_strength = picked
        if pupil_seed is not None and pupil_seed > 0:
            pupil_r, estimated = float(pupil_seed), False
        else:
            pupil_r, estimated = _estimate_pupil_in_disk(luminance, cx, cy, iris_r, pupil_iris_ratio)
        inv = 1.0 / scale
        cx_full, cy_full = cx * inv, cy * inv
        iris_r_full, pupil_r_full = iris_r * inv, pupil_r * inv
        confidence = float(np.clip(0.30 + edge_strength / 70.0, 0.30, 0.95))
        candidate_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(candidate_mask, (int(cx_full), int(cy_full)), int(iris_r_full), 255, 2)
        return IrisDiskEstimate(
            center_x=int(cx_full),
            center_y=int(cy_full),
            iris_radius=float(iris_r_full),
            pupil_radius=float(pupil_r_full),
            confidence=confidence,
            method="iris_disk_edge",
            pupil_estimated=estimated,
            candidate_mask=candidate_mask,
        )

    img_cx, img_cy = sw / 2.0, sh / 2.0
    best = None
    best_score = -float("inf")
    for cx, cy, radius in candidates.values():
        inside = _inside_mean(luminance, cx, cy, radius)
        support, outside = _surround_stats(luminance, cx, cy, radius, inside, surround_contrast * 0.5)
        contrast = outside - inside
        # 虹膜与巩膜的强反差是区分「眼睛」与「眉毛/睫毛/皮肤暗块」的关键：
        # 眉毛被皮肤包围时反差≈0，虹膜被巩膜包围时反差很大。
        if contrast < surround_contrast or support < surround_support_min:
            continue
        # 偏白巩膜占比作为加分项（不作硬门槛，避免暗光巩膜被误杀）
        sclera = _sclera_fraction(hsv, cx, cy, radius, sclera_v_min, sclera_s_max)
        # 双侧巩膜合理性：眉毛/睫毛两侧都被皮肤包围、无白巩膜，可据此拒绝
        bilateral = _bilateral_sclera_fraction(hsv, cx, cy, radius, sclera_v_min, sclera_s_max)
        if sclera_bilateral_min > 0.0 and bilateral < sclera_bilateral_min:
            continue

        # 多证据（与巩膜正交）：局部边界跃升 + 暗核致密度，压低锁在眉毛上的大圈。
        # 默认权重 0 时分数与历史一致；调高即启用。
        local_edge = (
            _local_edge_support(luminance, cx, cy, radius, local_edge_min_step)
            if local_edge_weight
            else 0.0
        )
        compact = (
            _dark_core_compactness(luminance, cx, cy, radius)
            if compactness_weight
            else 0.0
        )

        darkness = 1.0 - min(inside / 255.0, 1.0)
        dist_norm = ((cx - img_cx) ** 2 + (cy - img_cy) ** 2) ** 0.5 / max(sw, sh)
        score = (
            min(contrast / 80.0, 1.0) * 2.2
            + support * 1.2
            + darkness * 0.8
            + sclera * sclera_global_weight
            + bilateral * bilateral_weight
            + local_edge * local_edge_weight
            + compact * compactness_weight
            - dist_norm * 0.5
        )
        if score > best_score:
            best_score = score
            best = (cx, cy, radius, support, sclera)

    if best is None:
        return None

    cx, cy, iris_r, support, sclera = best
    pupil_r, estimated = _estimate_pupil_in_disk(luminance, cx, cy, iris_r, pupil_iris_ratio)

    inv = 1.0 / scale
    cx_full = cx * inv
    cy_full = cy * inv
    iris_r_full = iris_r * inv
    pupil_r_full = pupil_r * inv
    confidence = float(np.clip(0.30 + support * 0.35 + sclera * 0.6, 0.30, 0.95))

    candidate_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(candidate_mask, (int(cx_full), int(cy_full)), int(iris_r_full), 255, 2)

    return IrisDiskEstimate(
        center_x=int(cx_full),
        center_y=int(cy_full),
        iris_radius=float(iris_r_full),
        pupil_radius=float(pupil_r_full),
        confidence=confidence,
        method="iris_disk",
        pupil_estimated=estimated,
        candidate_mask=candidate_mask,
    )


def build_ring_from_disk(
    image_shape: Tuple[int, int],
    disk: IrisDiskEstimate,
    inner_iris_ratio: float = 0.40,
    outer_iris_ratio: float = 0.85,
) -> IrisRingEstimate:
    """以黑盘直检结果生成取色环带：内缘扣掉瞳孔，外缘留在虹膜中段。"""
    h, w = image_shape[:2]
    min_dim = min(h, w)
    iris_outer_r = disk.iris_radius

    inner_r = max(disk.pupil_radius * 1.08, iris_outer_r * inner_iris_ratio)
    outer_r = min(iris_outer_r * outer_iris_ratio, iris_outer_r - max(min_dim * 0.01, 2.0))
    if outer_r <= inner_r + 3:
        inner_r = min(disk.pupil_radius * 1.08, iris_outer_r - 6)
        outer_r = iris_outer_r - 3

    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = disk.center_x, disk.center_y
    cv2.circle(mask, (cx, cy), int(max(outer_r, 1)), 255, -1)
    cv2.circle(mask, (cx, cy), int(max(inner_r, 1)), 0, -1)
    sample_count = int(np.count_nonzero(mask))
    return IrisRingEstimate(
        mask=mask,
        outer_radius=float(outer_r),
        inner_radius=float(inner_r),
        sample_pixel_count=sample_count,
        iris_outer_method=disk.method,
        iris_confidence=disk.confidence,
    )
