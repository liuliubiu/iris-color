"""巩膜参考白平衡：环带采样 + 线性 RGB 对角增益（von Kries）。

将巩膜中位数映射到可配置目标近白，缓解机型 AWB / 整体曝光差异对虹膜 Lab 的影响。
定位仍用原图；校正只作用于取色用工作图副本。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from app.services.scope_field import ScopeField


@dataclass
class ScleraRef:
    """巩膜参考色（工作图坐标下采样）。"""

    median_bgr: Tuple[float, float, float]
    pixel_count: int
    clipped_ratio: float


@dataclass
class ScleraWbResult:
    """巩膜白平衡一次尝试的结果。"""

    applied: bool
    reason: str
    pixel_count: int = 0
    clipped_ratio: float = 0.0
    median_bgr: Optional[Tuple[float, float, float]] = None
    gains_bgr: Optional[Tuple[float, float, float]] = None
    gain_clamped: bool = False


def _srgb_u8_to_linear(u8: np.ndarray) -> np.ndarray:
    """uint8 sRGB → 线性光 [0,1]。"""
    c = np.clip(u8.astype(np.float64) / 255.0, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb_u8(linear: np.ndarray) -> np.ndarray:
    """线性光 [0,1] → uint8 sRGB。"""
    c = np.clip(linear, 0.0, 1.0)
    srgb = np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)
    return np.clip(np.rint(srgb * 255.0), 0, 255).astype(np.uint8)


def sample_sclera_reference(
    image_bgr: np.ndarray,
    center: Tuple[float, float],
    iris_r: float,
    *,
    scope: Optional[ScopeField] = None,
    inner_ratio: float = 1.08,
    outer_ratio: float = 1.28,
    v_min: float = 150.0,
    s_max: float = 60.0,
    highlight_v: float = 245.0,
    vessel_rg_max: float = 35.0,
    vessel_rb_max: float = 40.0,
    min_pixels: int = 80,
    clip_channel_min: float = 250.0,
) -> Optional[ScleraRef]:
    """
    在虹膜外侧环带采样偏白巩膜像素，返回 BGR 中位数参考色。

    失败（像素不足等）返回 None。
    """
    if iris_r <= 1.0:
        return None

    h, w = image_bgr.shape[:2]
    cx, cy = float(center[0]), float(center[1])
    r_in = float(iris_r) * float(inner_ratio)
    r_out = float(iris_r) * float(outer_ratio)
    if r_out <= r_in + 1.0:
        return None

    yy, xx = np.ogrid[:h, :w]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    annulus = (dist2 >= r_in * r_in) & (dist2 <= r_out * r_out)

    if scope is not None:
        sx, sy, sr = scope.center_x, scope.center_y, scope.radius
        # 略内收，避开镜筒边缘暗圈
        scope_r = max(sr * 0.96, 1.0)
        annulus &= (xx - sx) ** 2 + (yy - sy) ** 2 <= scope_r * scope_r

    if not np.any(annulus):
        return None

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]
    b = image_bgr[:, :, 0].astype(np.float32)
    g = image_bgr[:, :, 1].astype(np.float32)
    r = image_bgr[:, :, 2].astype(np.float32)

    whitish = (
        annulus
        & (v_ch >= v_min)
        & (v_ch < highlight_v)
        & (s_ch <= s_max)
        & ((r - g) < vessel_rg_max)
        & ((r - b) < vessel_rb_max)
    )
    count = int(whitish.sum())
    if count < min_pixels:
        return None

    pixels = image_bgr[whitish].astype(np.float64)
    median = np.median(pixels, axis=0)
    clipped = float(np.mean(np.any(pixels >= clip_channel_min, axis=1)))
    return ScleraRef(
        median_bgr=(float(median[0]), float(median[1]), float(median[2])),
        pixel_count=count,
        clipped_ratio=clipped,
    )


def compute_von_kries_gains(
    median_bgr: Sequence[float],
    target_srgb: Sequence[float],
    *,
    gain_min: float = 0.5,
    gain_max: float = 2.0,
    eps: float = 1e-4,
) -> Tuple[Tuple[float, float, float], bool]:
    """
    由巩膜中位数与目标 sRGB 计算 BGR 对角增益（线性空间）。

    target_srgb 为 R,G,B 顺序、范围 [0,1]。
    返回 (gains_bgr, gain_clamped)。
    """
    med_u8 = np.array(
        [[median_bgr[2], median_bgr[1], median_bgr[0]]], dtype=np.float64
    )  # RGB uint8-like
    med_lin = _srgb_u8_to_linear(med_u8)[0]
    tgt = np.clip(np.asarray(target_srgb, dtype=np.float64), 0.0, 1.0)
    # 目标亦按 sRGB 曲线理解
    tgt_u8 = np.clip(np.rint(tgt * 255.0), 0, 255)
    tgt_lin = _srgb_u8_to_linear(tgt_u8.reshape(1, 3))[0]

    raw = tgt_lin / np.maximum(med_lin, eps)
    clamped = np.clip(raw, gain_min, gain_max)
    was_clamped = bool(np.any(np.abs(clamped - raw) > 1e-9))
    # RGB gains → BGR order
    gains_bgr = (float(clamped[2]), float(clamped[1]), float(clamped[0]))
    return gains_bgr, was_clamped


def apply_sclera_white_balance(
    image_bgr: np.ndarray,
    gains_bgr: Sequence[float],
) -> np.ndarray:
    """对整图做线性空间对角增益，返回新的 uint8 BGR 图。"""
    linear = _srgb_u8_to_linear(image_bgr)
    gains = np.asarray(gains_bgr, dtype=np.float64).reshape(1, 1, 3)
    corrected = linear * gains
    return _linear_to_srgb_u8(corrected)


def correct_with_sclera_wb(
    image_bgr: np.ndarray,
    center: Tuple[float, float],
    iris_r: float,
    cfg: dict,
    *,
    scope: Optional[ScopeField] = None,
) -> Tuple[np.ndarray, ScleraWbResult]:
    """
    尝试巩膜白平衡。成功则返回校正图；失败则返回原图与原因。

    cfg 对应 eye_closeup.sclera_wb。
    """
    if not cfg.get("enabled", False):
        return image_bgr, ScleraWbResult(applied=False, reason="disabled")

    ref = sample_sclera_reference(
        image_bgr,
        center,
        iris_r,
        scope=scope,
        inner_ratio=float(cfg.get("inner_ratio", 1.08)),
        outer_ratio=float(cfg.get("outer_ratio", 1.28)),
        v_min=float(cfg.get("v_min", 150.0)),
        s_max=float(cfg.get("s_max", 60.0)),
        highlight_v=float(cfg.get("highlight_v", 245.0)),
        vessel_rg_max=float(cfg.get("vessel_rg_max", 35.0)),
        vessel_rb_max=float(cfg.get("vessel_rb_max", 40.0)),
        min_pixels=int(cfg.get("min_pixels", 80)),
        clip_channel_min=float(cfg.get("clip_channel_min", 250.0)),
    )
    if ref is None:
        return image_bgr, ScleraWbResult(applied=False, reason="insufficient_sclera_pixels")

    max_clip = float(cfg.get("max_clipped_ratio", 0.35))
    if ref.clipped_ratio > max_clip:
        return image_bgr, ScleraWbResult(
            applied=False,
            reason="sclera_clipped",
            pixel_count=ref.pixel_count,
            clipped_ratio=ref.clipped_ratio,
            median_bgr=ref.median_bgr,
        )

    # 中位数任一通道已贴顶，增益不可靠
    if max(ref.median_bgr) >= float(cfg.get("clip_channel_min", 250.0)):
        return image_bgr, ScleraWbResult(
            applied=False,
            reason="sclera_median_clipped",
            pixel_count=ref.pixel_count,
            clipped_ratio=ref.clipped_ratio,
            median_bgr=ref.median_bgr,
        )

    target = cfg.get("target_srgb", [0.92, 0.92, 0.90])
    if not isinstance(target, (list, tuple)) or len(target) != 3:
        target = [0.92, 0.92, 0.90]

    gains, gain_clamped = compute_von_kries_gains(
        ref.median_bgr,
        target,
        gain_min=float(cfg.get("gain_min", 0.5)),
        gain_max=float(cfg.get("gain_max", 2.0)),
    )
    corrected = apply_sclera_white_balance(image_bgr, gains)
    return corrected, ScleraWbResult(
        applied=True,
        reason="ok",
        pixel_count=ref.pixel_count,
        clipped_ratio=ref.clipped_ratio,
        median_bgr=ref.median_bgr,
        gains_bgr=gains,
        gain_clamped=gain_clamped,
    )
