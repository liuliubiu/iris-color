"""调试用可视化：各阶段区域叠加图。"""

import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import colorspacious
import cv2
import numpy as np

from app.services.color import (
    classify_iris_color,
    extract_iris_lab_median,
    linear_to_srgb,
    srgb_to_linear,
)
from app.services.grade import get_grade_boundaries, map_l_star_to_grade
from app.services.iris_detect import IrisDetectionResult
from app.services.pipeline import AnalysisPipelineResult


def _encode_jpeg(image_bgr: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("encode_failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _draw_center_roi(image_bgr: np.ndarray, ratio: float) -> np.ndarray:
    """标注瞳孔搜索 ROI（黄色框）。"""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    rw, rh = int(w * ratio), int(h * ratio)
    x1, y1 = (w - rw) // 2, (h - rh) // 2
    cv2.rectangle(out, (x1, y1), (x1 + rw, y1 + rh), (0, 255, 255), 2)
    cv2.putText(out, "pupil search ROI", (x1 + 4, max(y1 - 6, 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_pupil_localization(
    image_bgr: np.ndarray,
    detection: IrisDetectionResult,
    center_roi_ratio: float = 0.85,
) -> np.ndarray:
    """
    图 1：瞳孔定位
    - 黄框：搜索 ROI
    - 蓝圆：检测到的瞳孔
    - 红十字：瞳孔中心
    """
    out = _draw_center_roi(image_bgr, center_roi_ratio)
    if detection.pupil_center is None:
        cv2.putText(out, "pupil NOT detected", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        return out

    cx, cy = detection.pupil_center
    pr = detection.pupil_radius or 5
    estimated = (detection.pupil_method or "").endswith("estimated")
    pupil_color = (0, 165, 255) if estimated else (255, 120, 0)
    cv2.circle(out, (cx, cy), int(pr), pupil_color, 2)
    cv2.drawMarker(out, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
    pupil_label = f"pupil r={pr:.1f}"
    if estimated:
        pupil_label += " (estimated)"
    if detection.pupil_confidence is not None:
        pupil_label += f" conf={detection.pupil_confidence:.2f}"
    cv2.putText(out, pupil_label, (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, pupil_color, 1, cv2.LINE_AA)
    cv2.putText(out, f"method={detection.method}", (12, out.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_pupil_candidates(
    image_bgr: np.ndarray,
    detection: IrisDetectionResult,
) -> np.ndarray:
    """图 1a：瞳孔候选暗区，用于判断阈值是否把虹膜也吞进去。"""
    out = image_bgr.copy()
    candidate_mask = detection.candidate_mask
    if candidate_mask is None:
        cv2.putText(out, "candidate mask unavailable", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        return out

    overlay = out.copy()
    mask = candidate_mask > 0
    overlay[mask] = (overlay[mask] * 0.35 + np.array([0, 255, 255]) * 0.65).astype(np.uint8)
    out = cv2.addWeighted(overlay, 0.65, out, 0.35, 0)
    cv2.putText(
        out,
        f"yellow=dark candidates count={detection.candidate_count or 0}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def draw_iris_ring(
    image_bgr: np.ndarray,
    detection: IrisDetectionResult,
) -> np.ndarray:
    """
    图 2：虹膜环带
    - 绿圆：外缘（采样区外边界）
    - 红圆：内缘（排除瞳孔）
    - 青色半透明：环带区域
    """
    out = image_bgr.copy()
    cx, cy = detection.center
    inner_r = int(detection.inner_radius or detection.pupil_radius or 5)
    outer_r = int(detection.outer_radius or detection.radius)

    overlay = out.copy()
    ring_mask = detection.mask > 0
    overlay[ring_mask] = (overlay[ring_mask] * 0.45 + np.array([255, 255, 0]) * 0.55).astype(np.uint8)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)

    cv2.circle(out, (cx, cy), outer_r, (0, 220, 0), 2)
    cv2.circle(out, (cx, cy), inner_r, (0, 0, 255), 2)
    method = detection.iris_outer_method or "unknown"
    conf = detection.iris_confidence
    label = f"green=outer red=inner method={method}"
    if conf is not None:
        label += f" conf={conf:.2f}"
    cv2.putText(out, label, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_highlight_rejection(
    image_bgr: np.ndarray,
    sampling,
) -> np.ndarray:
    """
    图 3：去高光
    - 环带内保留像素：原色
    - 环带内高光像素：红色半透明覆盖
    - 环带外：变暗
    """
    out = image_bgr.copy().astype(np.float32)
    dim = ~sampling.ring
    out[dim] *= 0.25
    out = out.astype(np.uint8)

    highlight_overlay = out.copy()
    rejected = sampling.highlight_in_ring | sampling.bright_in_ring | sampling.dark_in_ring
    highlight_overlay[rejected] = (
        highlight_overlay[rejected] * 0.3 + np.array([0, 0, 255]) * 0.7
    ).astype(np.uint8)
    out = cv2.addWeighted(highlight_overlay, 0.7, out, 0.3, 0)
    cv2.putText(out, "red=rejected highlight/bright/dark in ring", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_valid_samples(
    image_bgr: np.ndarray,
    sampling,
) -> np.ndarray:
    """
    图 4：最终参与 Lab 统计的像素（绿色高亮）
    """
    out = image_bgr.copy().astype(np.float32) * 0.2
    out[sampling.valid] = image_bgr[sampling.valid].astype(np.float32)
    out = out.astype(np.uint8)
    count = int(sampling.valid.sum())
    cv2.putText(out, f"valid sample pixels: {count}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def draw_sclera_samples(
    image_bgr: np.ndarray,
    pipeline: AnalysisPipelineResult,
) -> np.ndarray:
    """
    图 6：巩膜参考采样区
    - 品红高亮：参与巩膜参考色统计的像素
    - 黄圆：巩膜采样环带内/外边界
    - 顶部文字：状态 + 巩膜 Lab + 增益
    """
    out = image_bgr.copy()
    sclera = pipeline.sclera
    if sclera is None:
        cv2.putText(out, "sclera normalization disabled", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        return out

    cx, cy = pipeline.detection.center
    if sclera.outer_radius > 0:
        cv2.circle(out, (cx, cy), int(sclera.inner_radius), (0, 255, 255), 2)
        cv2.circle(out, (cx, cy), int(sclera.outer_radius), (0, 255, 255), 2)

    if sclera.mask is not None:
        overlay = out.copy()
        mask = sclera.mask > 0
        overlay[mask] = (overlay[mask] * 0.35 + np.array([255, 0, 255]) * 0.65).astype(np.uint8)
        out = cv2.addWeighted(overlay, 0.65, out, 0.35, 0)

    status = pipeline.sclera_status
    color = (0, 255, 0) if status == "applied" else (0, 0, 255)
    cv2.putText(out, f"sclera status={status} pixels={sclera.pixel_count} "
                     f"clipped={sclera.clipped_ratio:.2f}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    if sclera.lab is not None:
        line = f"sclera Lab=({sclera.lab[0]:.1f}, {sclera.lab[1]:.1f}, {sclera.lab[2]:.1f})"
        if pipeline.sclera_gains is not None:
            g = pipeline.sclera_gains
            line += f" gains=({g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f})"
        cv2.putText(out, line, (12, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _lab_to_srgb01(lab: Tuple[float, float, float]) -> np.ndarray:
    srgb = colorspacious.cspace_convert(
        [float(lab[0]), float(lab[1]), float(lab[2])], "CIELab", "sRGB1"
    )
    return np.clip(np.asarray(srgb, dtype=np.float64), 0.0, 1.0)


def _lab_to_hex(lab: Tuple[float, float, float]) -> str:
    srgb = _lab_to_srgb01(lab)
    return "#{:02x}{:02x}{:02x}".format(
        int(round(srgb[0] * 255)),
        int(round(srgb[1] * 255)),
        int(round(srgb[2] * 255)),
    )


def _lab_to_bgr_u8(lab: Tuple[float, float, float]) -> Tuple[int, int, int]:
    srgb = _lab_to_srgb01(lab)
    return (
        int(round(srgb[2] * 255)),
        int(round(srgb[1] * 255)),
        int(round(srgb[0] * 255)),
    )


def _pack_side_metrics(
    lab,
    grade: int,
    confidence: float,
    iris_color,
) -> dict:
    lab_t = (float(lab.L), float(lab.a), float(lab.b))
    return {
        "lab": {"L": round(lab_t[0], 2), "a": round(lab_t[1], 2), "b": round(lab_t[2], 2)},
        "hex": _lab_to_hex(lab_t),
        "grade": int(grade),
        "confidence": float(confidence),
        "iris_color": {
            "code": iris_color.code,
            "label": iris_color.label,
            "confidence": iris_color.confidence,
            "hue": iris_color.hue,
            "depth": iris_color.depth,
        },
    }


def compute_color_correction_compare(
    pipeline: AnalysisPipelineResult,
    config: Optional[dict],
    highlight_v: int,
) -> dict:
    """
    同一份检测/采样下计算调色前（基线）与调色后数值，供调试台对比。

    未启用或未成功应用巩膜校正时，两侧数值相同，applied=false。
    """
    cfg = config or {}
    eye_cfg = cfg.get("eye_closeup", {})
    mad_trim = float(eye_cfg.get("color_trim_mad", 2.5)) if pipeline.scope is not None else 0.0
    sample_cap = int(eye_cfg.get("color_sample_cap", 20000))

    after = _pack_side_metrics(
        pipeline.lab,
        pipeline.grade.grade,
        pipeline.grade.confidence,
        pipeline.iris_color,
    )

    applied = pipeline.sclera_status == "applied" and pipeline.sclera_gains is not None
    if applied:
        base_lab = extract_iris_lab_median(
            pipeline.work_image,
            pipeline.detection.mask,
            highlight_v,
            sample_cap=sample_cap,
            masks=pipeline.sampling,
            mad_trim=mad_trim,
            channel_gains=None,
        )
        boundaries = get_grade_boundaries(cfg) if cfg else [55, 45, 29, 19]
        base_grade = map_l_star_to_grade(base_lab.L, boundaries)
        base_color = classify_iris_color(base_lab, cfg)
        before = _pack_side_metrics(
            base_lab, base_grade.grade, base_grade.confidence, base_color
        )
    else:
        before = after

    sclera = pipeline.sclera
    return {
        "applied": applied,
        "status": pipeline.sclera_status,
        "before": before,
        "after": after,
        "delta": {
            "L": round(after["lab"]["L"] - before["lab"]["L"], 2),
            "a": round(after["lab"]["a"] - before["lab"]["a"], 2),
            "b": round(after["lab"]["b"] - before["lab"]["b"], 2),
            "grade_changed": before["grade"] != after["grade"],
            "color_changed": before["iris_color"]["code"] != after["iris_color"]["code"],
        },
        "gains": [round(float(g), 4) for g in pipeline.sclera_gains]
        if pipeline.sclera_gains is not None
        else None,
        "sclera_lab": {
            "L": round(sclera.lab[0], 2),
            "a": round(sclera.lab[1], 2),
            "b": round(sclera.lab[2], 2),
        }
        if sclera is not None and sclera.lab is not None
        else None,
    }


def _apply_gains_bgr(
    image_bgr: np.ndarray,
    gains: np.ndarray,
) -> np.ndarray:
    """整图线性 RGB 对角增益（仅可视化；与取色路径一致，不含瞳孔黑偏置）。"""
    rgb = image_bgr[:, :, ::-1].astype(np.float64) / 255.0
    linear = srgb_to_linear(rgb)
    linear = np.clip(linear * np.asarray(gains, dtype=np.float64).reshape(1, 1, 3), 0.0, 1.0)
    srgb = linear_to_srgb(linear)
    return (np.clip(srgb[:, :, ::-1], 0.0, 1.0) * 255.0).astype(np.uint8)


def _iris_crop(image_bgr: np.ndarray, pipeline: AnalysisPipelineResult, pad_scale: float = 1.35):
    h, w = image_bgr.shape[:2]
    cx, cy = pipeline.detection.center
    r = float(pipeline.detection.outer_radius or pipeline.detection.radius or min(h, w) * 0.25)
    half = int(max(r * pad_scale, 40))
    x1 = max(int(cx) - half, 0)
    x2 = min(int(cx) + half, w)
    y1 = max(int(cy) - half, 0)
    y2 = min(int(cy) + half, h)
    return image_bgr[y1:y2, x1:x2].copy()


def _panel_with_swatch(
    crop_bgr: np.ndarray,
    side: dict,
    title: str,
    panel_w: int = 420,
    panel_h: int = 520,
) -> np.ndarray:
    """单侧对比面板：虹膜裁切 + Lab 色块 + 数值。"""
    panel = np.full((panel_h, panel_w, 3), 28, dtype=np.uint8)
    cv2.putText(panel, title, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    crop_area_h, crop_area_w = 300, panel_w - 32
    if crop_bgr.size > 0:
        ch, cw = crop_bgr.shape[:2]
        scale = min(crop_area_w / max(cw, 1), crop_area_h / max(ch, 1))
        nw, nh = max(int(cw * scale), 1), max(int(ch * scale), 1)
        resized = cv2.resize(crop_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        ox = 16 + (crop_area_w - nw) // 2
        oy = 48 + (crop_area_h - nh) // 2
        panel[oy : oy + nh, ox : ox + nw] = resized

    lab = side["lab"]
    swatch = _lab_to_bgr_u8((lab["L"], lab["a"], lab["b"]))
    y0 = 360
    cv2.rectangle(panel, (16, y0), (panel_w - 16, y0 + 56), swatch, -1)
    cv2.rectangle(panel, (16, y0), (panel_w - 16, y0 + 56), (200, 200, 200), 1)

    lines = [
        f"Lab=({lab['L']:.1f}, {lab['a']:.1f}, {lab['b']:.1f})",
        f"Grade {side['grade']}  {side['iris_color']['code']}",
        f"hex {side['hex']}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            panel,
            line,
            (16, y0 + 84 + i * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    return panel


def draw_sclera_before_after(
    pipeline: AnalysisPipelineResult,
    compare: dict,
) -> np.ndarray:
    """
    图 7：巩膜调色前 / 调色后对比
    - 左：原图虹膜裁切 + 基线 Lab 色块
    - 右：增益校正后裁切 + 校正 Lab 色块
    """
    before_img = pipeline.work_image
    if compare.get("applied") and pipeline.sclera_gains is not None:
        after_img = _apply_gains_bgr(pipeline.work_image, pipeline.sclera_gains)
    else:
        after_img = before_img

    left = _panel_with_swatch(
        _iris_crop(before_img, pipeline),
        compare["before"],
        "BEFORE (raw)",
    )
    right = _panel_with_swatch(
        _iris_crop(after_img, pipeline),
        compare["after"],
        "AFTER (sclera norm)" if compare.get("applied") else "AFTER (= BEFORE)",
    )
    gap = np.full((left.shape[0], 12, 3), 18, dtype=np.uint8)
    canvas = np.hstack([left, gap, right])

    footer_h = 56
    footer = np.full((footer_h, canvas.shape[1], 3), 18, dtype=np.uint8)
    status = compare.get("status", "?")
    color = (0, 220, 0) if compare.get("applied") else (0, 160, 255)
    delta = compare.get("delta") or {}
    line1 = f"status={status}  dL={delta.get('L', 0):+.2f}  da={delta.get('a', 0):+.2f}  db={delta.get('b', 0):+.2f}"
    cv2.putText(footer, line1, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    gains = compare.get("gains")
    sclera_lab = compare.get("sclera_lab")
    parts = []
    if gains is not None:
        parts.append(f"gains=({gains[0]:.3f}, {gains[1]:.3f}, {gains[2]:.3f})")
    if sclera_lab is not None:
        parts.append(
            f"sclera Lab=({sclera_lab['L']:.1f}, {sclera_lab['a']:.1f}, {sclera_lab['b']:.1f})"
        )
    if parts:
        cv2.putText(
            footer,
            "  ".join(parts),
            (16, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
    return np.vstack([canvas, footer])


def _shrink_to_max_dim(image_bgr: np.ndarray, max_dim: int) -> np.ndarray:
    """最长边超过 max_dim 时等比缩小（仅用于展示/编码，坐标语义不变）。"""
    if max_dim <= 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image_bgr
    scale = max_dim / longest
    return cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def build_debug_images(
    pipeline: AnalysisPipelineResult,
    eye_closeup_cfg: dict,
    max_dim: int = 1200,
    config: Optional[dict] = None,
    highlight_v: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """生成全部调试叠加图。

    在工作图（已裁剪/降采样）上绘制——detection/sampling 均为工作图坐标；
    输出前统一压到 max_dim 最长边，避免全分辨率拷贝与超大 base64。
    """
    image_bgr = pipeline.work_image
    images = {
        "01_pupil_candidates": draw_pupil_candidates(image_bgr, pipeline.detection),
        "01_pupil_localization": draw_pupil_localization(
            image_bgr,
            pipeline.detection,
            center_roi_ratio=eye_closeup_cfg.get("center_roi_ratio", 0.85),
        ),
        "02_iris_ring": draw_iris_ring(image_bgr, pipeline.detection),
        "03_highlight_rejection": draw_highlight_rejection(image_bgr, pipeline.sampling),
        "04_valid_samples": draw_valid_samples(image_bgr, pipeline.sampling),
        "05_ring_mask_only": cv2.applyColorMap(pipeline.detection.mask, cv2.COLORMAP_JET),
    }
    if pipeline.sclera is not None:
        images["06_sclera_samples"] = draw_sclera_samples(image_bgr, pipeline)

    hv = int(highlight_v if highlight_v is not None else 240)
    compare = compute_color_correction_compare(pipeline, config, hv)
    images["07_sclera_before_after"] = draw_sclera_before_after(pipeline, compare)
    return {name: _shrink_to_max_dim(img, max_dim) for name, img in images.items()}


def build_debug_metrics(
    pipeline: AnalysisPipelineResult,
    highlight_v: int,
    config: Optional[dict] = None,
) -> dict:
    """数值指标，便于对照图检查。坐标/半径均换算回原图坐标系。"""
    det = pipeline.detection
    smp = pipeline.sampling
    transform = pipeline.transform

    def _pt(point):
        if point is None:
            return None
        x, y = transform.to_original_xy(float(point[0]), float(point[1]))
        return [int(round(x)), int(round(y))]

    def _len(value):
        if value is None:
            return None
        return round(transform.to_original_len(float(value)), 1)

    pupil_center = _pt(det.pupil_center)
    scope = pipeline.scope
    compare = compute_color_correction_compare(pipeline, config, highlight_v)
    return {
        "detection_method": det.method,
        "detection_mode": pipeline.detection_mode,
        "manual_adjusted": det.method == "manual_adjustment",
        "manual_params": {
            "center_x": pupil_center[0] if pupil_center else None,
            "center_y": pupil_center[1] if pupil_center else None,
            "pupil_radius": _len(det.pupil_radius),
            "inner_radius": _len(det.inner_radius),
            "outer_radius": _len(det.outer_radius),
        } if det.method == "manual_adjustment" else None,
        "pupil_center": pupil_center,
        "pupil_radius": _len(det.pupil_radius),
        "inner_radius": _len(det.inner_radius),
        "outer_radius": _len(det.outer_radius),
        "scope_field": {
            "center": _pt((scope.center_x, scope.center_y)),
            "radius": _len(scope.radius),
        } if scope is not None else None,
        "work_scale": round(transform.scale, 4),
        "ring_pixel_count": int(smp.ring.sum()),
        "highlight_rejected_count": int(smp.highlight_in_ring.sum()),
        "bright_rejected_count": int(smp.bright_in_ring.sum()),
        "dark_rejected_count": int(smp.dark_in_ring.sum()),
        "valid_sample_count": int(smp.valid.sum()),
        "highlight_v_threshold": highlight_v,
        "pupil_method": det.pupil_method,
        "iris_outer_method": det.iris_outer_method,
        "pupil_confidence": det.pupil_confidence,
        "iris_confidence": det.iris_confidence,
        "candidate_count": det.candidate_count,
        "blur_score": round(pipeline.quality.blur_score, 2),
        "overexposed_ratio": round(pipeline.quality.overexposed_ratio, 4),
        "lab": {
            "L": round(pipeline.lab.L, 2),
            "a": round(pipeline.lab.a, 2),
            "b": round(pipeline.lab.b, 2),
        },
        "iris_color": {
            "code": pipeline.iris_color.code,
            "label": pipeline.iris_color.label,
            "confidence": pipeline.iris_color.confidence,
            "reason": pipeline.iris_color.reason,
            "hue": pipeline.iris_color.hue,
            "depth": pipeline.iris_color.depth,
        },
        "grade": pipeline.grade.grade,
        "confidence": pipeline.grade.confidence,
        "sclera_normalization": {
            "status": pipeline.sclera_status,
            "applied": pipeline.sclera_status == "applied",
            "lab": {
                "L": round(pipeline.sclera.lab[0], 2),
                "a": round(pipeline.sclera.lab[1], 2),
                "b": round(pipeline.sclera.lab[2], 2),
            } if pipeline.sclera is not None and pipeline.sclera.lab is not None else None,
            "gains": [round(float(g), 4) for g in pipeline.sclera_gains]
            if pipeline.sclera_gains is not None else None,
            "pixel_count": pipeline.sclera.pixel_count if pipeline.sclera is not None else 0,
            "clipped_ratio": pipeline.sclera.clipped_ratio if pipeline.sclera is not None else 0.0,
        },
        "color_correction": compare,
    }


def save_debug_run(
    output_root: Path,
    image_bgr: np.ndarray,
    images: Dict[str, np.ndarray],
    metrics: dict,
) -> Tuple[str, Path]:
    """保存到 debug_output/{run_id}/，返回 run_id 与目录。"""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(run_dir / "00_original.jpg"), image_bgr)
    for name, img in images.items():
        cv2.imwrite(str(run_dir / f"{name}.jpg"), img)

    import json
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return run_id, run_dir


def images_to_base64(images: Dict[str, np.ndarray]) -> Dict[str, str]:
    """供 API 直接返回 base64（可选，体积较大）。"""
    return {k: _encode_jpeg(v) for k, v in images.items()}
