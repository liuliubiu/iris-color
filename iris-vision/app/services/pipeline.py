"""分析流水线：供正式接口与调试接口共用。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from app.services.color import (
    IrisColorResult,
    LabResult,
    SamplingMasks,
    classify_iris_color,
    compute_sampling_masks,
    extract_iris_lab_median,
)
from app.services.grade import GradeResult, grade_from_l_star
from app.services.iris_detect import IrisDetectionResult, detect_iris_ring_mask
from app.services.quality import QualityResult, check_quality


@dataclass
class AnalysisPipelineResult:
    """完整分析结果（含调试中间数据）。"""

    quality: QualityResult
    detection: IrisDetectionResult
    sampling: SamplingMasks
    lab: LabResult
    grade: GradeResult
    iris_color: IrisColorResult
    detection_mode: str


class AnalysisError(Exception):
    """分析失败，携带 error code 与可选 quality。"""

    def __init__(self, code: str, quality: Optional[QualityResult] = None):
        self.code = code
        self.quality = quality
        super().__init__(code)


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_analysis(
    image_bgr,
    config: dict,
    config_path: Path,
    *,
    skip_quality: bool = False,
) -> AnalysisPipelineResult:
    """执行质量检测 → 定位 → 取色 → 分档。"""
    quality_cfg = config.get("quality", {})
    ring_cfg = config.get("iris_ring", {})
    detection_cfg = config.get("detection", {})
    eye_closeup_cfg = config.get("eye_closeup", {})
    highlight_v = config.get("highlight_v_threshold", 240)
    detection_mode = detection_cfg.get("mode", "eye_closeup")

    quality = check_quality(
        image_bgr,
        blur_threshold=quality_cfg.get("blur_threshold", 15.0),
        overexposed_ratio_max=quality_cfg.get("overexposed_ratio_max", 0.15),
        detection_mode=detection_mode,
    )
    if not skip_quality and not quality.passed:
        raise AnalysisError("quality_check_failed", quality)

    detection = detect_iris_ring_mask(
        image_bgr,
        mode=detection_mode,
        inner_ratio=ring_cfg.get("inner_ratio", 0.30),
        outer_ratio=ring_cfg.get("outer_ratio", 0.80),
        eye_closeup_cfg=eye_closeup_cfg,
    )
    if detection is None:
        raise AnalysisError("no_iris_detected", quality)

    min_pixels = quality_cfg.get("min_sample_pixels", 50)
    if detection.sample_pixel_count < min_pixels:
        raise AnalysisError("insufficient_iris_samples", quality)

    sampling = compute_sampling_masks(image_bgr, detection.mask, highlight_v)
    if int(sampling.valid.sum()) < min_pixels:
        raise AnalysisError("no_valid_pixels_after_highlight_removal", quality)

    lab = extract_iris_lab_median(image_bgr, detection.mask, highlight_v)
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
    )
