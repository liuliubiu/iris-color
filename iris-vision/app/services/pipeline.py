"""分析流水线：供正式接口与调试接口共用。

流程：预处理（视场裁剪+统一工作分辨率）→ 质量检测 → 定位 → 遮挡剔除 → 取色 → 分档。
内部全部在「工作图」坐标系执行；对外接口的坐标经 transform 换算回原图。
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from app.services.color import (
    IrisColorResult,
    LabResult,
    SamplingMasks,
    classify_iris_color,
    compute_sampling_masks,
    extract_iris_lab_median,
    filter_ring_mask_sectors,
)
from app.services.grade import GradeResult, grade_from_l_star
from app.services.iris_detect import IrisDetectionResult, build_manual_iris_detection, detect_iris_ring_mask
from app.services.quality import QualityResult, check_quality
from app.services.scope_field import ImageTransform, ScopeField, preprocess_capture


@dataclass
class AnalysisPipelineResult:
    """完整分析结果（含调试中间数据）。detection/sampling 均为工作图坐标。"""

    quality: QualityResult
    detection: IrisDetectionResult
    sampling: SamplingMasks
    lab: LabResult
    grade: GradeResult
    iris_color: IrisColorResult
    detection_mode: str
    work_image: np.ndarray
    transform: ImageTransform
    scope: Optional[ScopeField] = None


class AnalysisError(Exception):
    """分析失败，携带 error code 与可选 quality。"""

    def __init__(self, code: str, quality: Optional[QualityResult] = None):
        self.code = code
        self.quality = quality
        super().__init__(code)


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _manual_to_working(params: dict, transform: ImageTransform) -> dict:
    """人工调整参数由原图坐标换算到工作图坐标。"""
    if transform.identity:
        return dict(params)
    out = dict(params)
    try:
        cx, cy = transform.to_working_xy(float(params["center_x"]), float(params["center_y"]))
        out["center_x"], out["center_y"] = cx, cy
        for key in ("pupil_radius", "inner_radius", "outer_radius"):
            out[key] = transform.to_working_len(float(params[key]))
    except (KeyError, TypeError, ValueError):
        return dict(params)
    return out


def run_analysis(
    image_bgr,
    config: dict,
    config_path: Path,
    *,
    skip_quality: bool = False,
    manual_detection: Optional[dict] = None,
    closeup_mode: str = "auto",
) -> AnalysisPipelineResult:
    """执行预处理 → 质量检测 → 定位 → 取色 → 分档。"""
    quality_cfg = config.get("quality", {})
    eye_closeup_cfg = config.get("eye_closeup", {})
    highlight_v = config.get("highlight_v_threshold", 240)
    detection_mode = "eye_closeup"

    # 镜筒特写：裁掉黑边并统一工作分辨率；普通图仅按需降采样
    pre = preprocess_capture(image_bgr, eye_closeup_cfg)
    work = pre.image
    transform = pre.transform
    scope = pre.scope

    quality = check_quality(
        work,
        blur_threshold=quality_cfg.get("blur_threshold", 15.0),
        overexposed_ratio_max=quality_cfg.get("overexposed_ratio_max", 0.15),
        scope=scope,
    )
    if not skip_quality and not quality.passed:
        raise AnalysisError("quality_check_failed", quality)

    if manual_detection is not None:
        detection = build_manual_iris_detection(
            work.shape, _manual_to_working(manual_detection, transform)
        )
        detection_mode = "manual_adjustment"
    else:
        detection = detect_iris_ring_mask(
            work,
            eye_closeup_cfg=eye_closeup_cfg,
            closeup_mode=closeup_mode,
            scope=scope,
        )
    if detection is None:
        raise AnalysisError("no_iris_detected", quality)

    # 环带扇区遮挡剔除：自动砍掉眼皮/睫毛/反光楔形区。
    # 仅镜筒特写启用（该类图上部遮挡是常态）；普通图与人工调整保持原行为。
    sectors_cfg = eye_closeup_cfg.get("ring_sectors", {})
    if (
        scope is not None
        and detection.method != "manual_adjustment"
        and sectors_cfg.get("enabled", True)
        and detection.sample_pixel_count > 0
    ):
        filtered_mask, kept = filter_ring_mask_sectors(
            work,
            detection.mask,
            detection.center,
            sector_count=int(sectors_cfg.get("count", 36)),
            v_dev_max=float(sectors_cfg.get("v_dev_max", 45.0)),
            min_keep_ratio=float(sectors_cfg.get("min_keep_ratio", 0.40)),
        )
        if kept > 0:
            detection = replace(
                detection,
                mask=filtered_mask,
                sample_pixel_count=int(np.count_nonzero(filtered_mask)),
            )

    min_pixels = quality_cfg.get("min_sample_pixels", 50)
    if detection.sample_pixel_count < min_pixels:
        raise AnalysisError("insufficient_iris_samples", quality)

    sampling = compute_sampling_masks(work, detection.mask, highlight_v)
    if int(sampling.valid.sum()) < min_pixels:
        raise AnalysisError("no_valid_pixels_after_highlight_removal", quality)

    color_sample_cap = int(eye_closeup_cfg.get("color_sample_cap", 20000))
    # MAD 鲁棒修剪仅对镜筒特写启用，避免改变普通图既有取色基线
    mad_trim = float(eye_closeup_cfg.get("color_trim_mad", 2.5)) if scope is not None else 0.0
    lab = extract_iris_lab_median(
        work,
        detection.mask,
        highlight_v,
        sample_cap=color_sample_cap,
        masks=sampling,
        mad_trim=mad_trim,
    )
    grade = grade_from_l_star(lab.L, config_path)
    iris_color = classify_iris_color(lab, config)

    return AnalysisPipelineResult(
        quality=quality,
        detection=detection,
        sampling=sampling,
        lab=lab,
        grade=grade,
        iris_color=iris_color,
        detection_mode=detection_mode,
        work_image=work,
        transform=transform,
        scope=scope,
    )
