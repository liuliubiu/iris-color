"""巩膜参考色提取与 von Kries 通道增益计算。

同一个人的巩膜（眼白）颜色基本恒定且接近中性白；不同设备白平衡/照度导致的
整图色偏与曝光偏移会同样体现在巩膜上。在虹膜外侧左右两个楔形环带内稳健采样
巩膜像素，求出其线性 RGB 中值，再算一组对角增益把巩膜"拉回"固定参考中性灰，
用同一组增益校正虹膜采样像素即可吸收设备间色差与照度差。
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import colorspacious
import cv2
import numpy as np

from app.services.color import linear_to_srgb, srgb_to_linear
from app.services.iris_detect import IrisDetectionResult
from app.services.scope_field import ScopeField


@dataclass
class ScleraReference:
    """巩膜参考色提取结果。"""

    ok: bool
    reason: str
    pixel_count: int = 0
    clipped_ratio: float = 0.0
    rgb_linear: Optional[np.ndarray] = None  # (3,) 线性 RGB 中值，0..1
    lab: Optional[Tuple[float, float, float]] = None
    mask: Optional[np.ndarray] = None  # 参与统计的像素（uint8，调试用）
    inner_radius: float = 0.0
    outer_radius: float = 0.0


def _y_from_l_star(l_star: float) -> float:
    """CIE L* → 相对亮度 Y（白点归一）。"""
    fy = (l_star + 16.0) / 116.0
    y = fy ** 3
    if y <= 0.008856:
        y = l_star / 903.3
    return y


def _linear_luminance(rgb: np.ndarray) -> float:
    """线性 RGB → 相对亮度 Y（Rec.709 系数）。"""
    return float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])


def extract_pupil_black_offset(
    image_bgr: np.ndarray,
    detection: IrisDetectionResult,
    cfg: dict,
) -> Optional[np.ndarray]:
    """
    以瞳孔为黑参考估计加性偏置（线性 RGB）。

    瞳孔本应接近纯黑；其实测亮度 ≈ 雾状眩光（对焦不实/镜头炫光的加性成分）
    + 相机底噪。取瞳孔内圈像素线性 RGB 的低分位数作为偏置估计，低分位数
    天然避开瞳孔内的镜面反光亮斑。
    """
    center = detection.pupil_center or detection.center
    pupil_r = float(detection.pupil_radius or 0.0)
    if pupil_r < 4.0:
        return None

    h, w = image_bgr.shape[:2]
    cx, cy = center
    r = pupil_r * float(cfg.get("pupil_radius_scale", 0.7))
    x1 = max(int(cx - r), 0)
    x2 = min(int(cx + r) + 1, w)
    y1 = max(int(cy - r), 0)
    y2 = min(int(cy + r) + 1, h)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None

    sub = image_bgr[y1:y2, x1:x2]
    xx = np.arange(x1, x2, dtype=np.float32) - float(cx)
    yy = np.arange(y1, y2, dtype=np.float32) - float(cy)
    inside = (xx[None, :] ** 2 + yy[:, None] ** 2) <= r * r
    if int(np.count_nonzero(inside)) < int(cfg.get("pupil_min_pixels", 50)):
        return None

    rgb = sub[inside][:, ::-1].astype(np.float64) / 255.0
    linear = srgb_to_linear(rgb)
    percentile = float(cfg.get("pupil_black_percentile", 25.0))
    offset = np.percentile(linear, percentile, axis=0)
    return np.clip(offset, 0.0, 1.0)


def extract_sclera_reference(
    image_bgr: np.ndarray,
    detection: IrisDetectionResult,
    scope: Optional[ScopeField],
    cfg: dict,
) -> ScleraReference:
    """
    在虹膜外侧环带的左右楔形区采样巩膜，返回稳健参考色。

    几何：以虹膜外径为基准取 [ring_inner_scale, ring_outer_scale] 环带，
    只保留左右 ±wedge_half_angle_deg 楔形（上下是眼皮/睫毛），并裁剪到
    镜筒视场圆内（视场边缘有渐晕，收缩到 scope_max_ratio）。

    像素过滤（防污染）：
    - 剔除削波/镜面反光：任一通道 >= clip_v（削波像素无色彩信息）
    - 剔除皮肤/睫毛/阴影：S > s_max 或 V < v_min
    - 剔除血管：R-G 差超过 rg_diff_max 的偏红像素
    最后按 V 做 MAD 修剪再取线性 RGB 各通道中位数。
    """
    h, w = image_bgr.shape[:2]
    cx, cy = detection.center
    iris_r = float(detection.outer_radius or detection.radius or 0.0)
    if iris_r <= 2.0:
        return ScleraReference(ok=False, reason="no_iris_radius")

    inner_r = iris_r * float(cfg.get("ring_inner_scale", 1.10))
    outer_r = iris_r * float(cfg.get("ring_outer_scale", 1.60))
    half_rad = np.deg2rad(float(cfg.get("wedge_half_angle_deg", 55.0)))

    x1 = max(int(cx - outer_r), 0)
    x2 = min(int(cx + outer_r) + 1, w)
    y1 = max(int(cy - outer_r), 0)
    y2 = min(int(cy + outer_r) + 1, h)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return ScleraReference(ok=False, reason="annulus_out_of_bounds")

    sub = image_bgr[y1:y2, x1:x2]
    xx = np.arange(x1, x2, dtype=np.float32) - float(cx)
    yy = np.arange(y1, y2, dtype=np.float32) - float(cy)
    dx = xx[None, :]
    dy = yy[:, None]
    dist2 = dx * dx + dy * dy

    geo = (dist2 >= inner_r * inner_r) & (dist2 <= outer_r * outer_r)
    angle = np.arctan2(np.broadcast_to(dy, geo.shape), np.broadcast_to(dx, geo.shape))
    abs_angle = np.abs(angle)
    geo &= (abs_angle <= half_rad) | (abs_angle >= np.pi - half_rad)

    if scope is not None:
        scope_max = float(cfg.get("scope_max_ratio", 0.92)) * float(scope.radius)
        sdx = xx[None, :] + float(cx) - float(scope.center_x)
        sdy = yy[:, None] + float(cy) - float(scope.center_y)
        geo &= (sdx * sdx + sdy * sdy) <= scope_max * scope_max

    if not np.any(geo):
        return ScleraReference(
            ok=False, reason="no_geometry_pixels", inner_radius=inner_r, outer_radius=outer_r
        )

    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    s_chan = hsv[:, :, 1].astype(np.float32)
    v_chan = hsv[:, :, 2].astype(np.float32)
    b16 = sub[:, :, 0].astype(np.int16)
    g16 = sub[:, :, 1].astype(np.int16)
    r16 = sub[:, :, 2].astype(np.int16)

    v_min = float(cfg.get("v_min", 100.0))
    s_max = float(cfg.get("s_max", 70.0))
    rg_diff_max = float(cfg.get("rg_diff_max", 30.0))
    clip_v = float(cfg.get("clip_v", 250.0))

    candidate = geo & (v_chan >= v_min) & (s_chan <= s_max) & ((r16 - g16) <= rg_diff_max)
    max_chan = np.maximum(np.maximum(b16, g16), r16).astype(np.float32)
    clipped = candidate & (max_chan >= clip_v)
    valid = candidate & ~clipped

    candidate_count = int(np.count_nonzero(candidate))
    valid_count = int(np.count_nonzero(valid))
    clipped_ratio = (
        float(np.count_nonzero(clipped)) / candidate_count if candidate_count > 0 else 0.0
    )

    min_pixels = int(cfg.get("min_pixels", 500))
    if valid_count < min_pixels:
        return ScleraReference(
            ok=False,
            reason="insufficient_sclera_pixels",
            pixel_count=valid_count,
            clipped_ratio=round(clipped_ratio, 4),
            inner_radius=inner_r,
            outer_radius=outer_r,
        )
    if clipped_ratio > float(cfg.get("clipped_ratio_max", 0.50)):
        return ScleraReference(
            ok=False,
            reason="sclera_overexposed",
            pixel_count=valid_count,
            clipped_ratio=round(clipped_ratio, 4),
            inner_radius=inner_r,
            outer_radius=outer_r,
        )

    def _drop_from_valid(keep: np.ndarray) -> None:
        nonlocal valid, valid_count
        ys_v, xs_v = np.nonzero(valid)
        drop = ~keep
        valid[ys_v[drop], xs_v[drop]] = False
        valid_count = int(np.count_nonzero(valid))

    # 左右楔形亮度差过大时只用更亮一侧：裂隙灯窄光束下常见单侧巩膜在阴影里，
    # 阴影侧会把参考拉低导致虹膜被过度提亮
    side_v_gap = float(cfg.get("side_v_gap", 0.0))
    if side_v_gap > 0 and valid_count >= 100:
        right_side = np.broadcast_to(dx, valid.shape) >= 0
        right_valid = valid & right_side
        left_valid = valid & ~right_side
        n_right = int(np.count_nonzero(right_valid))
        n_left = int(np.count_nonzero(left_valid))
        if n_right >= min_pixels and n_left >= min_pixels:
            v_right = float(np.median(v_chan[right_valid]))
            v_left = float(np.median(v_chan[left_valid]))
            if abs(v_right - v_left) > side_v_gap:
                brighter = right_valid if v_right > v_left else left_valid
                valid = brighter
                valid_count = int(np.count_nonzero(valid))

    # 只用较亮的巩膜像素做参考：阴影中的巩膜会拉低参考值，
    # 而「被照亮的巩膜」才与虹膜受到的照明对应
    bright_percentile = float(cfg.get("bright_percentile", 0.0))
    if bright_percentile > 0 and valid_count >= 100:
        v_vals = v_chan[valid]
        cutoff = float(np.percentile(v_vals, bright_percentile))
        keep = v_vals >= cutoff
        if int(keep.sum()) >= min_pixels:
            _drop_from_valid(keep)

    # 按 V 做 MAD 修剪，压掉残余的局部阴影/亮斑
    mad_trim = float(cfg.get("mad_trim", 2.5))
    if mad_trim > 0 and valid_count >= 100:
        v_vals = v_chan[valid]
        median_v = float(np.median(v_vals))
        mad = float(np.median(np.abs(v_vals - median_v)))
        if mad > 1e-6:
            keep = np.abs(v_vals - median_v) <= mad_trim * mad
            if int(keep.sum()) >= max(min_pixels, int(valid_count * 0.3)):
                _drop_from_valid(keep)

    pixels_bgr = sub[valid]
    sample_cap = int(cfg.get("sample_cap", 50000))
    if sample_cap and len(pixels_bgr) > sample_cap:
        rng = np.random.default_rng(12345)
        idx = rng.choice(len(pixels_bgr), size=sample_cap, replace=False)
        pixels_bgr = pixels_bgr[idx]

    rgb = pixels_bgr[:, ::-1].astype(np.float64) / 255.0
    rgb_linear_median = np.median(srgb_to_linear(rgb), axis=0)
    if float(np.min(rgb_linear_median)) <= 1e-6:
        return ScleraReference(
            ok=False,
            reason="sclera_too_dark",
            pixel_count=valid_count,
            clipped_ratio=round(clipped_ratio, 4),
            inner_radius=inner_r,
            outer_radius=outer_r,
        )

    srgb_median = linear_to_srgb(rgb_linear_median)
    lab = colorspacious.cspace_convert(srgb_median, "sRGB1", "CIELab")

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2][valid] = 255

    return ScleraReference(
        ok=True,
        reason="ok",
        pixel_count=valid_count,
        clipped_ratio=round(clipped_ratio, 4),
        rgb_linear=rgb_linear_median,
        lab=(float(lab[0]), float(lab[1]), float(lab[2])),
        mask=full_mask,
        inner_radius=inner_r,
        outer_radius=outer_r,
    )


def compute_channel_gains(
    reference: ScleraReference,
    cfg: dict,
    black_offset: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], str]:
    """
    由巩膜参考色计算线性 RGB 对角增益（von Kries）。

    black_offset 非空时按仿射模型标定：corrected = (pixel - offset) × gain，
    即先扣掉雾状眩光/底噪的加性偏置，再把「净巩膜色」拉到目标中性灰。

    - normalize_chroma + normalize_luminance：巩膜 → 目标中性灰（L*=target_l）
    - 仅 normalize_chroma：巩膜 → 与自身等亮度的中性灰（只去色偏）
    - 仅 normalize_luminance：各通道等比缩放到目标亮度（只归一曝光）

    增益超出 [gain_min, gain_max] 视为参考不可靠，放弃校正。
    """
    if not reference.ok or reference.rgb_linear is None:
        return None, reference.reason

    normalize_luminance = bool(cfg.get("normalize_luminance", True))
    normalize_chroma = bool(cfg.get("normalize_chroma", True))
    if not normalize_luminance and not normalize_chroma:
        return None, "normalization_disabled"

    rgb = reference.rgb_linear.copy()
    if black_offset is not None:
        rgb = rgb - black_offset
    if float(np.min(rgb)) <= 1e-4:
        return None, "sclera_too_dark"
    y_sclera = _linear_luminance(rgb)

    y_target = _y_from_l_star(float(cfg.get("target_l", 75.0)))

    # 色度增益：巩膜 → 与自身等亮度的中性灰（不改变整体曝光）
    chroma_gains = (y_sclera / rgb) if normalize_chroma else np.ones(3, dtype=np.float64)

    # 亮度增益：strength<1 做部分校正（巩膜受照程度本身有波动，全量校正
    # 会把巩膜采样噪声放大进虹膜 L*）；再截断到合理曝光差范围
    lum_scale = 1.0
    if normalize_luminance:
        strength = float(cfg.get("luminance_strength", 1.0))
        lum_scale = (y_target / y_sclera) ** strength
        lum_scale = float(np.clip(
            lum_scale,
            float(cfg.get("lum_gain_min", 0.5)),
            float(cfg.get("lum_gain_max", 2.0)),
        ))

    gains = chroma_gains * lum_scale

    gain_min = float(cfg.get("gain_min", 0.25))
    gain_max = float(cfg.get("gain_max", 4.0))
    if float(np.min(gains)) < gain_min or float(np.max(gains)) > gain_max:
        return None, "gain_out_of_range"

    return gains.astype(np.float64), "ok"
