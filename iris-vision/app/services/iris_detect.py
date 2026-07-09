"""虹膜环带 mask 生成：眼部特写（镜筒图）专用。

实拍图均为「眼睛占满画面」的镜筒特写，不含全脸/眉毛，因此这里只保留
眼部特写两阶段定位（精定位 + 黑盘直检），不再依赖 MediaPipe 全脸 landmark。
"""

from dataclasses import dataclass, replace
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
from app.services.scope_field import ScopeField


def _downscale_for_detection(
    image_bgr: np.ndarray, target_min_dim: int
) -> Tuple[np.ndarray, float]:
    """统一降采样到目标最短边，返回 (小图, scale)。已足够小则原样返回 scale=1。"""
    h, w = image_bgr.shape[:2]
    min_dim = min(h, w)
    if target_min_dim <= 0 or min_dim <= target_min_dim:
        return image_bgr, 1.0
    scale = float(target_min_dim) / float(min_dim)
    small = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return small, scale


def _ring_mask(
    shape: Tuple[int, int], cx: int, cy: int, inner_r: float, outer_r: float
) -> Tuple[np.ndarray, int]:
    """在给定尺寸上重建虹膜环带 mask。"""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), int(max(outer_r, 1)), 255, -1)
    cv2.circle(mask, (cx, cy), int(max(inner_r, 1)), 0, -1)
    return mask, int(np.count_nonzero(mask))


def _scale_detection_to_full(
    result: "IrisDetectionResult", inv: float, full_shape: Tuple[int, int]
) -> "IrisDetectionResult":
    """把降采样空间的检测结果缩放回全分辨率，并在全分辨率重建 mask。"""
    if result is None:
        return None
    if abs(inv - 1.0) < 1e-6:
        return result

    h, w = full_shape[:2]

    def _s(v):
        return None if v is None else float(v) * inv

    cx, cy = result.center
    full_cx, full_cy = int(round(cx * inv)), int(round(cy * inv))
    inner_r = _s(result.inner_radius) or 0.0
    outer_r = _s(result.outer_radius) or _s(result.radius) or 1.0
    mask, sample_count = _ring_mask(full_shape, full_cx, full_cy, inner_r, outer_r)

    pupil_center = None
    if result.pupil_center is not None:
        pcx, pcy = result.pupil_center
        pupil_center = (int(round(pcx * inv)), int(round(pcy * inv)))

    candidate_mask = result.candidate_mask
    if candidate_mask is not None:
        candidate_mask = cv2.resize(
            candidate_mask, (w, h), interpolation=cv2.INTER_NEAREST
        )

    return replace(
        result,
        mask=mask,
        center=(full_cx, full_cy),
        radius=_s(result.radius),
        sample_pixel_count=sample_count,
        pupil_center=pupil_center,
        pupil_radius=_s(result.pupil_radius),
        inner_radius=inner_r,
        outer_radius=outer_r,
        candidate_mask=candidate_mask,
    )


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


def _detect_precise(
    detect_image: np.ndarray,
    full_shape: Tuple[int, int],
    eye_cfg: dict,
    spec_mask: Optional[np.ndarray] = None,
) -> Optional[IrisDetectionResult]:
    """阶段A：瞳孔精定位 + 径向梯度虹膜外缘（适合清晰分离图）。"""
    pupil = detect_pupil(
        detect_image,
        center_roi_ratio=eye_cfg.get("center_roi_ratio", 0.85),
        dark_percentile=eye_cfg.get("pupil_dark_percentile", 12.0),
        spec_mask=spec_mask,
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
        spec_mask=spec_mask,
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
    scope: Optional[ScopeField] = None,
) -> Optional[IrisDetectionResult]:
    """阶段B：黑盘直检（适合瞳孔与虹膜融成一团黑的实拍图）。"""
    disk_cfg = eye_cfg.get("disk", {})
    scope_cfg = eye_cfg.get("scope_field", {})
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
        sclera_bilateral_min=disk_cfg.get("sclera_bilateral_min", 0.0),
        target_min_dim=int(disk_cfg.get("target_min_dim", 700)),
        scope=scope,
        scope_iris_min_ratio=float(scope_cfg.get("iris_r_min_ratio", 0.35)),
        scope_iris_max_ratio=float(scope_cfg.get("iris_r_max_ratio", 0.75)),
        limbus_edge_scale=float(scope_cfg.get("limbus_edge_scale", 1.06)),
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


def _detect_modes_core(
    img: np.ndarray,
    spec_mask: Optional[np.ndarray],
    eye_cfg: dict,
    mode: str,
    prefer_disk: bool = False,
    scope: Optional[ScopeField] = None,
) -> Optional[IrisDetectionResult]:
    """在给定图像上按子策略跑精定位/黑盘直检。结果坐标为该图局部坐标。

    prefer_disk: 大图（已降采样的实拍照片）经验上黑盘直检更稳，优先 disk；
    小清晰图沿用精定位优先。
    scope: 镜筒视场圆（该图局部坐标），作为黑盘直检的尺度/位置先验。
    """
    shape = img.shape
    if mode == "precise":
        return _detect_precise(img, shape, eye_cfg, spec_mask)
    if mode == "rough":
        return _detect_rough(img, shape, eye_cfg, scope)

    # auto
    if prefer_disk or scope is not None:
        rough = _detect_rough(img, shape, eye_cfg, scope)
        if rough is not None and rough.sample_pixel_count > 0:
            return rough
        return _detect_precise(img, shape, eye_cfg, spec_mask)

    min_confidence = eye_cfg.get("auto_precise_min_confidence", 0.5)
    precise = _detect_precise(img, shape, eye_cfg, spec_mask)
    if precise is not None and _precise_acceptable(precise, min_confidence):
        return precise

    rough = _detect_rough(img, shape, eye_cfg)
    if rough is not None:
        return rough
    return precise


def _detect_from_eye_closeup(
    image_bgr: np.ndarray,
    eye_cfg: dict,
    mode: str = "auto",
    scope: Optional[ScopeField] = None,
) -> Optional[IrisDetectionResult]:
    """
    眼部特写定位：统一降采样 → 镜面反光修复 → 两阶段定位，最后缩放回全分辨率。

    mode:
      - auto: 精定位置信度达标用 A，否则回退黑盘直检 B
      - precise: 强制阶段A（清晰精定位）
      - rough: 强制阶段B（实拍黑盘直检）
    scope: 镜筒视场圆（image_bgr 坐标），存在时约束黑盘直检的尺度/位置。
    """
    target = int(eye_cfg.get("detect_target_min_dim", 900))
    small, scale = _downscale_for_detection(image_bgr, target)
    inv = 1.0 / scale

    scope_small = scope
    if scope is not None and scale < 1.0:
        scope_small = ScopeField(
            center_x=scope.center_x * scale,
            center_y=scope.center_y * scale,
            radius=scope.radius * scale,
            bright_ratio=scope.bright_ratio,
        )

    specular_cfg = eye_cfg.get("specular", {})
    if specular_cfg.get("enabled", True):
        detect_image, spec_mask = suppress_specular(
            small,
            v_threshold=specular_cfg.get("v_threshold", 230),
            sat_threshold=specular_cfg.get("sat_threshold", 90),
            max_area_ratio=specular_cfg.get("max_area_ratio", 0.04),
            dilate_px=specular_cfg.get("dilate_px", 3),
        )
    else:
        detect_image, spec_mask = small, None

    # 反光跳过默认关闭：经回归会扰动清晰小图（误把高光当反光），且对实测反光图无增益
    detect_spec = spec_mask if specular_cfg.get("skip_rays", False) else None

    # 经降采样的大图（实拍照片）优先黑盘直检；小清晰图沿用精定位优先
    prefer_disk = scale < 1.0
    result = _detect_modes_core(
        detect_image, detect_spec, eye_cfg, mode, prefer_disk, scope_small
    )
    return _scale_detection_to_full(result, inv, image_bgr.shape)


def detect_iris_ring_mask(
    image_bgr: np.ndarray,
    eye_closeup_cfg: Optional[dict] = None,
    closeup_mode: str = "auto",
    scope: Optional[ScopeField] = None,
) -> Optional[IrisDetectionResult]:
    """
    检测虹膜环带 mask（眼部特写）。

    closeup_mode（眼部特写子策略）:
      - auto: 精定位置信度达标用精定位，否则回退黑盘直检
      - precise: 强制清晰精定位
      - rough: 强制实拍黑盘直检

    scope: 镜筒视场圆（image_bgr 坐标），由预处理提供。
    """
    eye_cfg = eye_closeup_cfg or {}
    return _detect_from_eye_closeup(image_bgr, eye_cfg, mode=closeup_mode, scope=scope)
