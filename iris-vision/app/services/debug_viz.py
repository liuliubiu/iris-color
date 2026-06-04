"""调试用可视化：各阶段区域叠加图。"""

import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

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


def build_debug_images(
    image_bgr: np.ndarray,
    pipeline: AnalysisPipelineResult,
    eye_closeup_cfg: dict,
) -> Dict[str, np.ndarray]:
    """生成全部调试叠加图。"""
    return {
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


def build_debug_metrics(pipeline: AnalysisPipelineResult, highlight_v: int) -> dict:
    """数值指标，便于对照图检查。"""
    det = pipeline.detection
    smp = pipeline.sampling
    return {
        "detection_method": det.method,
        "detection_mode": pipeline.detection_mode,
        "manual_adjusted": det.method == "manual_adjustment",
        "manual_params": {
            "center_x": det.pupil_center[0] if det.pupil_center else None,
            "center_y": det.pupil_center[1] if det.pupil_center else None,
            "pupil_radius": det.pupil_radius,
            "inner_radius": det.inner_radius,
            "outer_radius": det.outer_radius,
        } if det.method == "manual_adjustment" else None,
        "pupil_center": list(det.pupil_center) if det.pupil_center else None,
        "pupil_radius": det.pupil_radius,
        "inner_radius": det.inner_radius,
        "outer_radius": det.outer_radius,
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
        },
        "grade": pipeline.grade.grade,
        "confidence": pipeline.grade.confidence,
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
