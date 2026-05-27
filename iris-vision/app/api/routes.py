"""FastAPI 路由定义。"""

from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException

from app.models.schemas import (
    AnalysisResponse,
    ErrorResponse,
    HealthResponse,
    LabValues,
    QualityInfo,
)
from app.services.color import extract_iris_lab_median
from app.services.grade import grade_from_l_star
from app.services.iris_detect import detect_iris_ring_mask
from app.services.quality import check_quality

router = APIRouter()

# 配置文件路径（相对于 iris-vision 根目录）
CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "grade_thresholds.yaml"


def _load_yaml_config() -> dict:
    import yaml

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _decode_image(content: bytes) -> np.ndarray:
    """将上传字节解码为 OpenCV BGR 图像。"""
    arr = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="invalid_image_format")
    return image


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """健康检查。"""
    return HealthResponse()


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def analyze(file: UploadFile = File(...)) -> AnalysisResponse:
    """
    上传眼部照片，返回 CIELAB 与 Grade。

    处理链：质量检测 → 虹膜定位 → 取色 → 分档
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file_must_be_image")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="empty_file")

    config = _load_yaml_config()
    quality_cfg = config.get("quality", {})
    ring_cfg = config.get("iris_ring", {})
    detection_cfg = config.get("detection", {})
    eye_closeup_cfg = config.get("eye_closeup", {})
    highlight_v = config.get("highlight_v_threshold", 240)
    detection_mode = detection_cfg.get("mode", "eye_closeup")

    image_bgr = _decode_image(content)

    # 1. 质量检测
    quality = check_quality(
        image_bgr,
        blur_threshold=quality_cfg.get("blur_threshold", 15.0),
        overexposed_ratio_max=quality_cfg.get("overexposed_ratio_max", 0.15),
        detection_mode=detection_mode,
    )

    if not quality.passed:
        raise HTTPException(
            status_code=422,
            detail={
                "success": False,
                "error": "quality_check_failed",
                "quality": {
                    "blur_score": quality.blur_score,
                    "overexposed_ratio": quality.overexposed_ratio,
                    "eye_open": quality.eye_open,
                    "sample_pixel_count": 0,
                    "issues": quality.issues,
                },
            },
        )

    # 2. 虹膜定位（默认：眼部特写，不依赖全脸）
    detection = detect_iris_ring_mask(
        image_bgr,
        mode=detection_mode,
        inner_ratio=ring_cfg.get("inner_ratio", 0.30),
        outer_ratio=ring_cfg.get("outer_ratio", 0.80),
        eye_closeup_cfg=eye_closeup_cfg,
    )
    if detection is None:
        raise HTTPException(
            status_code=400,
            detail="no_iris_detected",
        )

    min_pixels = quality_cfg.get("min_sample_pixels", 50)
    if detection.sample_pixel_count < min_pixels:
        raise HTTPException(status_code=422, detail="insufficient_iris_samples")

    # 3. 取色 + Lab
    try:
        lab_result = extract_iris_lab_median(
            image_bgr,
            detection.mask,
            highlight_v_threshold=highlight_v,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # 4. 分档
    grade_result = grade_from_l_star(lab_result.L, CONFIG_PATH)

    return AnalysisResponse(
        quality=QualityInfo(
            blur_score=round(quality.blur_score, 2),
            overexposed_ratio=round(quality.overexposed_ratio, 4),
            eye_open=quality.eye_open,
            sample_pixel_count=lab_result.sample_pixel_count,
            issues=[],
        ),
        lab=LabValues(
            L=round(lab_result.L, 2),
            a=round(lab_result.a, 2),
            b=round(lab_result.b, 2),
        ),
        grade=grade_result.grade,
        confidence=grade_result.confidence,
        detection_method=detection.method,
    )
