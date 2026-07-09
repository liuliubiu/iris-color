"""眼部特写图：基于瞳孔边界与虹膜外缘定位取色环带。"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from app.services.scope_field import ScopeField


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


def _fit_circle_kasa(xs: np.ndarray, ys: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Kasa 代数最小二乘圆拟合，返回 (cx, cy, r)。"""
    if len(xs) < 3:
        return None
    a_mat = np.column_stack([xs, ys, np.ones(len(xs))])
    b_vec = xs * xs + ys * ys
    try:
        sol, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx = float(sol[0]) / 2.0
    cy = float(sol[1]) / 2.0
    r_sq = float(sol[2]) + cx * cx + cy * cy
    if r_sq <= 0:
        return None
    return cx, cy, float(np.sqrt(r_sq))


def _dark_blob_seed(
    luminance: np.ndarray,
    scope_cx: float,
    scope_cy: float,
    scope_r: float,
) -> Optional[Tuple[float, float]]:
    """
    视场中心区（0.55r 内）暗连通域的质心，作为虹膜中心种子。

    镜筒对准眼球拍摄，瞳孔（或瞳孔+暗虹膜核）总在视场中心附近；
    限制在中心区可避免被下方限带阴影、睫毛丛等外围暗结构拉偏。
    """
    h, w = luminance.shape[:2]
    interior = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(interior, (int(scope_cx), int(scope_cy)), max(int(scope_r * 0.55), 1), 255, -1)
    vals = luminance[interior > 0]
    if vals.size == 0:
        return None
    thr = float(np.percentile(vals, 25))
    dark = ((luminance <= thr) & (interior > 0)).astype(np.uint8) * 255
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, open_kernel)
    close_size = max(int(scope_r * 0.12) | 1, 9)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, close_kernel)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if num <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy = centroids[idx]
    return float(cx), float(cy)


def _detect_disk_by_rays(
    luminance: np.ndarray,
    scope_cx: float,
    scope_cy: float,
    scope_r: float,
    r_min_ratio: float,
    r_max_ratio: float,
) -> Optional[Tuple[float, float, float, float]]:
    """
    镜筒特写专用：种子中心 + 径向亮度跃升点 + 圆拟合迭代。

    虹膜边缘在这类图上是软梯度（角膜缘暗环 → 亮巩膜），Hough 常常漏检；
    直接沿下方 240° 射线找「暗 → 亮」最大跃升点再做稳健圆拟合，
    对上眼皮遮挡与局部反光都不敏感。返回 (cx, cy, r, confidence)。
    """
    seed = _dark_blob_seed(luminance, scope_cx, scope_cy, scope_r)
    if seed is None:
        return None
    cx, cy = seed

    angles = np.deg2rad(np.linspace(-20.0, 200.0, 48, endpoint=False))
    r_lo = scope_r * r_min_ratio * 0.75
    r_hi = scope_r * r_max_ratio * 1.10
    result = None

    for _ in range(3):
        pts_x, pts_y, radii = [], [], []
        radial = np.linspace(r_lo, r_hi, 90)
        for angle in angles:
            xs = cx + np.cos(angle) * radial
            ys = cy + np.sin(angle) * radial
            values = _sample_luminance_at(luminance, xs, ys)
            smooth = _smooth_profile(values, 7)
            gradients = np.diff(smooth)
            if len(gradients) < 10:
                continue
            # 全局最大跃升：可能停在虹膜内部纹理上（半径略小），但采样环
            # 仍全落在虹膜内，对取色安全；宁可偏小也不越过巩膜
            idx = int(np.argmax(gradients))
            if idx < 3 or idx > len(gradients) - 4:
                continue
            local_gain = float(gradients[idx])
            total_gain = float(
                smooth[min(idx + 5, len(smooth) - 1)] - smooth[max(idx - 5, 0)]
            )
            if local_gain < 2.0 and total_gain < 8.0:
                continue
            pts_x.append(float(xs[idx]))
            pts_y.append(float(ys[idx]))
            radii.append(float(radial[idx]))

        if len(radii) < 12:
            break

        radii_arr = np.asarray(radii, dtype=np.float32)
        median_r = float(np.median(radii_arr))
        mad = float(np.median(np.abs(radii_arr - median_r))) or 1.0
        inlier = np.abs(radii_arr - median_r) <= 2.0 * mad
        if int(inlier.sum()) < 10:
            break

        fit = _fit_circle_kasa(
            np.asarray(pts_x, dtype=np.float64)[inlier],
            np.asarray(pts_y, dtype=np.float64)[inlier],
        )
        if fit is None:
            break
        fx, fy, fr = fit

        # 拟合结果必须落在视场先验内
        center_dist = ((fx - scope_cx) ** 2 + (fy - scope_cy) ** 2) ** 0.5
        if (
            fr < scope_r * r_min_ratio * 0.9
            or fr > scope_r * r_max_ratio * 1.05
            or center_dist > scope_r * 0.75
        ):
            break

        support = float(inlier.sum()) / float(len(angles))
        confidence = float(np.clip(0.35 + support * 0.6, 0.35, 0.95))
        result = (fx, fy, fr, confidence)
        cx, cy = fx, fy
        r_lo = fr * 0.75
        r_hi = fr * 1.25

    if result is None:
        return None
    # 终检：边界内外须有「暗 → 亮」跃升，拒绝锁在睫毛暗斑上的假圆
    fx, fy, fr, confidence = result
    support, contrast = _boundary_stats(luminance, fx, fy, fr, 5.0)
    if support < 0.45 or contrast < 6.0:
        return None
    return result


def _boundary_stats(
    luminance: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    margin: float,
) -> Tuple[float, float]:
    """
    圆边界两侧采样：返回 (外亮于内的角度占比, 内外亮度差中位数)。

    镜筒实拍的虹膜常是亮棕色，「盘内均值暗于外侧」不成立；但虹膜边缘的
    角膜缘暗环 → 巩膜的亮度跃升非常稳定。在 0.90r / 1.12r 两侧采样、
    只统计下方 240° 弧段（上方被眼皮遮挡），比全盘均值可靠得多。
    """
    angles = np.deg2rad(np.linspace(-30.0, 210.0, 24, endpoint=False))
    in_vals = _sample_luminance_at(
        luminance, cx + np.cos(angles) * radius * 0.90, cy + np.sin(angles) * radius * 0.90
    )
    out_vals = _sample_luminance_at(
        luminance, cx + np.cos(angles) * radius * 1.12, cy + np.sin(angles) * radius * 1.12
    )
    diff = out_vals - in_vals
    support = float(np.mean(diff > margin))
    contrast = float(np.median(diff))
    return support, contrast


def _refine_disk_radius(
    luminance: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
) -> float:
    """
    沿下方 240° 弧段的射线把半径吸附到最大亮度跃升处（角膜缘 → 巩膜）。

    Hough 半径步长较粗，直接用会把环带压进巩膜或缩进虹膜；
    按射线梯度中位数精修可稳定贴合真实虹膜外缘。
    """
    angles = np.deg2rad(np.linspace(-30.0, 210.0, 36, endpoint=False))
    radial = np.linspace(radius * 0.78, radius * 1.22, 56)
    refined = []
    for angle in angles:
        xs = cx + np.cos(angle) * radial
        ys = cy + np.sin(angle) * radial
        values = _sample_luminance_at(luminance, xs, ys)
        smooth = _smooth_profile(values, 5)
        gradients = np.diff(smooth)
        if len(gradients) == 0:
            continue
        idx = int(np.argmax(gradients))
        if gradients[idx] < 2.0 or idx < 2 or idx > len(gradients) - 3:
            continue
        refined.append(float(radial[idx]))

    if len(refined) < 10:
        return radius
    arr = np.asarray(refined, dtype=np.float32)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median))) or 1.0
    inliers = arr[np.abs(arr - median) <= 2.5 * mad]
    if len(inliers) < 8:
        return radius
    return float(np.median(inliers))


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
    scope: Optional[ScopeField] = None,
    scope_iris_min_ratio: float = 0.35,
    scope_iris_max_ratio: float = 0.75,
) -> Optional[IrisDiskEstimate]:
    """
    黑盘直检：直接用 Hough 找虹膜/巩膜的圆形边界，再用「内暗外亮」验证。

    适用于瞳孔与虹膜融成一团黑、且与睫毛/眉毛/眼窝阴影连成一片的实拍图——
    此时暗区连通域不再是圆，必须靠圆形边界（梯度）而非暗区面积定位。

    scope 非空（镜筒特写）时启用视场先验：虹膜半径限制在视场半径的
    [scope_iris_min_ratio, scope_iris_max_ratio] 区间、圆心须在视场内、
    环绕验证只看下方 240° 弧段（上方是眼皮）。
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

    scope_cx = scope_cy = scope_r = None
    if scope is not None:
        scope_cx = scope.center_x * scale
        scope_cy = scope.center_y * scale
        scope_r = scope.radius * scale

    if scope_r is not None:
        # 镜筒特写主路径：种子中心 + 径向跃升 + 圆拟合（对软边缘/遮挡更稳）
        ray_fit = _detect_disk_by_rays(
            luminance, scope_cx, scope_cy, scope_r,
            scope_iris_min_ratio, scope_iris_max_ratio,
        )
        if ray_fit is not None:
            cx, cy, iris_r, confidence = ray_fit
            pupil_r, estimated = _estimate_pupil_in_disk(
                luminance, cx, cy, iris_r, pupil_iris_ratio
            )
            inv = 1.0 / scale
            candidate_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(
                candidate_mask,
                (int(cx * inv), int(cy * inv)),
                int(iris_r * inv),
                255,
                2,
            )
            return IrisDiskEstimate(
                center_x=int(cx * inv),
                center_y=int(cy * inv),
                iris_radius=float(iris_r * inv),
                pupil_radius=float(pupil_r * inv),
                confidence=confidence,
                method="iris_disk_rays",
                pupil_estimated=estimated,
                candidate_mask=candidate_mask,
            )
        # 兜底：Hough 候选 + 边界跃升验证，搜索半径仍受视场先验约束
        min_r = max(int(scope_r * scope_iris_min_ratio), 8)
        max_r = max(int(scope_r * scope_iris_max_ratio), min_r + 1)
    else:
        min_r = max(int(s_min_dim * min_radius_ratio), 8)
        max_r = max(int(s_min_dim * max_radius_ratio), min_r + 1)
    blurred = cv2.medianBlur(luminance, 5)

    # 多阈值收集候选并去重：严阈值给强边界，松阈值补遮挡/弱边界。
    # 镜筒图（有视场先验）三档全量收集后统一评分，避免小假圆挤掉被遮挡的真虹膜；
    # 普通图保留「凑够 2 个候选提前短路」，与既有回归基线一致。
    candidates: dict = {}
    for param2 in (30, 22, 16):
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_r,
            param1=50,
            param2=param2,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is not None:
            for circle in circles[0]:
                cx, cy, radius = float(circle[0]), float(circle[1]), float(circle[2])
                key = (round(cx / 12), round(cy / 12), round(radius / 12))
                candidates.setdefault(key, (cx, cy, radius))
        if scope is None and len(candidates) >= 2:
            break
    if not candidates:
        return None

    img_cx, img_cy = sw / 2.0, sh / 2.0
    best = None
    best_score = -float("inf")
    for cx, cy, radius in candidates.values():
        if scope_r is not None:
            # 镜筒特写：虹膜是亮棕色时「盘内暗于外侧」不成立，改用
            # 边界跃升验证（角膜缘暗环 → 巩膜），且只看下方 240° 弧段。
            center_dist = ((cx - scope_cx) ** 2 + (cy - scope_cy) ** 2) ** 0.5
            if center_dist > scope_r * 0.75:
                continue
            if center_dist + radius > scope_r * 1.05:
                continue
            support, contrast = _boundary_stats(luminance, cx, cy, radius, 5.0)
            if support < surround_support_min or contrast < 6.0:
                continue
            sclera = _sclera_fraction(hsv, cx, cy, radius, sclera_v_min, sclera_s_max)
            score = (
                support * 2.0
                + min(contrast / 40.0, 1.0) * 2.0
                + min(radius / scope_r, 1.0)
                + sclera * 0.6
                - (center_dist / scope_r) * 0.8
            )
            if score > best_score:
                best_score = score
                best = (cx, cy, radius, support, sclera)
            continue

        inside = _inside_mean(luminance, cx, cy, radius)
        support, outside = _surround_stats(
            luminance, cx, cy, radius, inside, surround_contrast * 0.5
        )
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

        darkness = 1.0 - min(inside / 255.0, 1.0)
        dist_norm = ((cx - img_cx) ** 2 + (cy - img_cy) ** 2) ** 0.5 / max(sw, sh)
        score = (
            min(contrast / 80.0, 1.0) * 2.2
            + support * 1.2
            + darkness * 0.8
            + sclera * 0.8
            + bilateral * 0.8
            - dist_norm * 0.5
        )
        if score > best_score:
            best_score = score
            best = (cx, cy, radius, support, sclera)

    if best is None:
        return None

    cx, cy, iris_r, support, sclera = best
    if scope_r is not None:
        # Hough 半径较粗，按下方弧段的亮度跃升精修外缘
        iris_r = _refine_disk_radius(luminance, cx, cy, iris_r)
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
