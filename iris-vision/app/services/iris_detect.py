"""虹膜环带 mask 生成：支持眼部特写与全脸两种模式。"""

from dataclasses import dataclass, replace
from typing import Mapping, Optional, Tuple

import cv2
import numpy as np

from app.services.eye_iris_detect import (
    _bilateral_sclera_fraction,
    build_refined_iris_ring,
    build_ring_from_disk,
    detect_iris_disk,
    detect_pupil,
    suppress_specular,
)
from app.services.eye_roi import EyePrior, detect_eye_prior
from app.services.face_landmarker import detect_face_landmarks
from app.services.scope_field import ScopeField

_LEFT_IRIS_CENTER = 468
_RIGHT_IRIS_CENTER = 473


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


def _offset_detection(
    result: "IrisDetectionResult", off_x: int, off_y: int, parent_shape: Tuple[int, int]
) -> "IrisDetectionResult":
    """把 ROI 局部坐标系的检测结果平移并嵌入父图尺寸（仍为降采样空间）。"""
    if result is None:
        return None
    h, w = parent_shape[:2]
    mh, mw = result.mask.shape[:2]
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[off_y : off_y + mh, off_x : off_x + mw] = result.mask
    candidate_mask = result.candidate_mask
    if candidate_mask is not None:
        cm = np.zeros((h, w), dtype=np.uint8)
        ch, cw = candidate_mask.shape[:2]
        cm[off_y : off_y + ch, off_x : off_x + cw] = candidate_mask
        candidate_mask = cm

    def _shift(point):
        if point is None:
            return None
        return (int(point[0]) + off_x, int(point[1]) + off_y)

    return replace(
        result,
        mask=full_mask,
        center=_shift(result.center),
        pupil_center=_shift(result.pupil_center),
        sample_pixel_count=int(np.count_nonzero(full_mask)),
        candidate_mask=candidate_mask,
    )


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


def _landmark_to_pixel(landmark, width: int, height: int) -> Tuple[int, int]:
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    return x, y


def _estimate_iris_radius(landmarks, center_idx: int, width: int, height: int) -> float:
    center = landmarks[center_idx]
    cx, cy = center.x * width, center.y * height
    iris_edge_indices = {
        _LEFT_IRIS_CENTER: [469, 470, 471, 472],
        _RIGHT_IRIS_CENTER: [474, 475, 476, 477],
    }
    edge_indices = iris_edge_indices.get(center_idx, [])
    if not edge_indices:
        return min(width, height) * 0.04

    distances = []
    for idx in edge_indices:
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        dx = lm.x * width - cx
        dy = lm.y * height - cy
        distances.append((dx * dx + dy * dy) ** 0.5)

    if not distances:
        return min(width, height) * 0.04
    return float(np.mean(distances))


def _detect_from_face_landmarks(
    image_bgr: np.ndarray,
    inner_ratio: float,
    outer_ratio: float,
) -> Optional[IrisDetectionResult]:
    """全脸模式：MediaPipe Face Landmarker + 虹膜 landmark。"""
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    landmarks = detect_face_landmarks(rgb)
    if landmarks is None:
        return None

    for center_idx, side in [(_RIGHT_IRIS_CENTER, "right"), (_LEFT_IRIS_CENTER, "left")]:
        if center_idx >= len(landmarks):
            continue

        center_lm = landmarks[center_idx]
        cx, cy = _landmark_to_pixel(center_lm, w, h)
        radius = _estimate_iris_radius(landmarks, center_idx, w, h)
        if radius < 5:
            continue

        inner_r = radius * inner_ratio
        outer_r = radius * outer_ratio
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), int(outer_r), 255, -1)
        cv2.circle(mask, (cx, cy), int(inner_r), 0, -1)
        sample_count = int(np.count_nonzero(mask))
        if sample_count < 10:
            continue

        return IrisDetectionResult(
            mask=mask,
            center=(cx, cy),
            radius=outer_r,
            eye_side=side,
            sample_pixel_count=sample_count,
            method="face_landmark",
            pupil_center=(cx, cy),
            pupil_radius=inner_r,
            inner_radius=inner_r,
            outer_radius=outer_r,
            pupil_confidence=0.7,
            iris_confidence=0.7,
            pupil_method="face_landmark",
            iris_outer_method="face_landmark",
            candidate_count=0,
        )
    return None


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


def _bilateral_sclera_value(
    image_bgr: np.ndarray, result: IrisDetectionResult, eye_cfg: dict
) -> float:
    """候选虹膜左右两侧白色巩膜占比的较小值（双侧都需有巩膜）。无半径时返回 1.0。"""
    if result is None:
        return 0.0
    outer_r = result.outer_radius or result.radius
    if not outer_r:
        return 1.0
    disk_cfg = eye_cfg.get("disk", {})
    cx, cy = result.center
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return _bilateral_sclera_fraction(
        hsv,
        float(cx),
        float(cy),
        float(outer_r),
        disk_cfg.get("sclera_v_min", 150.0),
        disk_cfg.get("sclera_s_max", 60.0),
    )


def _detect_modes_core(
    img: np.ndarray,
    spec_mask: Optional[np.ndarray],
    eye_cfg: dict,
    mode: str,
    prefer_disk: bool = False,
    scope: Optional[ScopeField] = None,
) -> Optional[IrisDetectionResult]:
    """在给定图像（全帧或 ROI）上按 mode 跑精定位/黑盘直检。结果坐标为该图局部坐标。

    prefer_disk: 大图（已降采样的实拍照片）经验上黑盘直检更稳，优先 disk；
    小清晰图沿用精定位优先。这与全量回归一致：所有大图基线均走 disk。
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


def _prior_disagrees(
    base: Optional[IrisDetectionResult], prior: EyePrior, eye_cfg: dict
) -> bool:
    """全帧定位中心与 MediaPipe 眼部先验明显不一致（疑似锁在眉毛/睫毛等画面干扰上）。"""
    if base is None:
        return False
    factor = float(eye_cfg.get("prior_disagree_factor", 1.0))
    if factor <= 0:
        return False
    bx, by = base.center
    dist = ((bx - prior.iris_cx) ** 2 + (by - prior.iris_cy) ** 2) ** 0.5
    return dist > factor * max(prior.iris_r, 1.0)


def _run_modes(
    detect_image: np.ndarray,
    spec_mask: Optional[np.ndarray],
    prior: Optional[EyePrior],
    eye_cfg: dict,
    mode: str,
    prefer_disk: bool = False,
    scope: Optional[ScopeField] = None,
) -> Optional[IrisDetectionResult]:
    """
    先走全帧逻辑；仅当全帧中心与眼部先验明显不一致（疑似锁在眉毛/睫毛上）时，
    才用 MediaPipe 先验裁 ROI 重定位抢救，且抢救结果须双侧巩膜更优才替换。

    先验与全帧一致时（WechatIMG1027 等）完全不介入，保证已验证结果不受影响；
    模型缺失/无脸时 prior=None，同样走原路径。
    """
    base = _detect_modes_core(detect_image, spec_mask, eye_cfg, mode, prefer_disk, scope)
    if mode == "precise" or prior is None or prior.roi is None:
        return base
    if not _prior_disagrees(base, prior, eye_cfg):
        return base

    x, y, rw, rh = prior.roi
    sub = detect_image[y : y + rh, x : x + rw]
    sub_spec = spec_mask[y : y + rh, x : x + rw] if spec_mask is not None else None
    res = _detect_modes_core(sub, sub_spec, eye_cfg, mode)
    if res is None or res.sample_pixel_count <= 0:
        return base
    cand = _offset_detection(res, x, y, detect_image.shape)

    suspect_min = float(eye_cfg.get("eyebrow_suspect_sclera_min", 0.06))
    base_sclera = _bilateral_sclera_value(detect_image, base, eye_cfg)
    cand_sclera = _bilateral_sclera_value(detect_image, cand, eye_cfg)
    if cand_sclera >= suspect_min and cand_sclera > base_sclera:
        return cand
    return base


def _detect_from_eye_closeup(
    image_bgr: np.ndarray,
    eye_cfg: dict,
    mode: str = "auto",
    scope: Optional[ScopeField] = None,
) -> Optional[IrisDetectionResult]:
    """
    眼部特写模式：统一降采样 → 镜面反光修复 → 眼部先验 ROI → 两阶段定位，最后缩放回全分辨率。

    mode:
      - auto: 精定位置信度达标用 A，否则回退黑盘直检 B
      - precise: 强制阶段A（清晰精定位）
      - rough: 强制阶段B（实拍黑盘直检）
    眼部先验（MediaPipe）缺席时优雅回退到纯几何启发式。
    scope: 镜筒视场圆（image_bgr 坐标），存在时跳过 MediaPipe 先验并约束黑盘直检。
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

    # 镜筒特写检不出人脸，MediaPipe 先验必然落空，直接跳过省一次模型推理
    prior = None
    if scope is None and eye_cfg.get("use_landmark_prior", True):
        prior = detect_eye_prior(
            detect_image, float(eye_cfg.get("landmark_roi_margin", 3.0))
        )

    # 经降采样的大图（实拍照片）优先黑盘直检；小清晰图沿用精定位优先
    prefer_disk = scale < 1.0
    result = _run_modes(
        detect_image, detect_spec, prior, eye_cfg, mode, prefer_disk, scope_small
    )
    return _scale_detection_to_full(result, inv, image_bgr.shape)


def detect_iris_ring_mask(
    image_bgr: np.ndarray,
    mode: str = "eye_closeup",
    inner_ratio: float = 0.30,
    outer_ratio: float = 0.80,
    eye_closeup_cfg: Optional[dict] = None,
    closeup_mode: str = "auto",
    scope: Optional[ScopeField] = None,
) -> Optional[IrisDetectionResult]:
    """
    检测虹膜环带 mask。

    mode（定位来源）:
      - eye_closeup: 默认，适用于「眼睛占满画面」的拍照/上传图
      - face: 全脸图，MediaPipe 定位
      - auto: 先 eye_closeup，失败再 face

    closeup_mode（眼部特写子策略）:
      - auto: 精定位置信度达标用精定位，否则回退黑盘直检
      - precise: 强制清晰精定位
      - rough: 强制实拍黑盘直检

    scope: 镜筒视场圆（image_bgr 坐标），由预处理提供；仅 eye_closeup 路径使用。
    """
    eye_cfg = eye_closeup_cfg or {}

    if mode == "eye_closeup":
        return _detect_from_eye_closeup(image_bgr, eye_cfg, mode=closeup_mode, scope=scope)
    if mode == "face":
        return _detect_from_face_landmarks(image_bgr, inner_ratio, outer_ratio)

    # auto
    result = _detect_from_eye_closeup(image_bgr, eye_cfg, mode=closeup_mode, scope=scope)
    if result is not None:
        return result
    return _detect_from_face_landmarks(image_bgr, inner_ratio, outer_ratio)
