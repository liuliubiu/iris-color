"""FastAPI 路由定义。"""

from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import (
    AnalysisResponse,
    ErrorResponse,
    HealthResponse,
    IrisColorInfo,
    LabValues,
    QualityInfo,
)
from app.services.pipeline import AnalysisError, load_config, run_analysis

router = APIRouter()

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "grade_thresholds.yaml"


def _decode_image(content: bytes) -> np.ndarray:
    arr = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="invalid_image_format")
    return image


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def analyze(file: UploadFile = File(...)) -> AnalysisResponse:
    """上传眼部特写，返回 CIELAB 与 Grade（无调试信息）。"""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file_must_be_image")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty_file")

    config = load_config(CONFIG_PATH)
    image_bgr = _decode_image(content)

    try:
        result = run_analysis(image_bgr, config, CONFIG_PATH)
    except AnalysisError as exc:
        if exc.code == "quality_check_failed" and exc.quality:
            raise HTTPException(
                status_code=422,
                detail={
                    "success": False,
                    "error": exc.code,
                    "quality": {
                        "blur_score": exc.quality.blur_score,
                        "overexposed_ratio": exc.quality.overexposed_ratio,
                        "eye_open": exc.quality.eye_open,
                        "sample_pixel_count": 0,
                        "issues": exc.quality.issues,
                    },
                },
            ) from exc
        status = 400 if exc.code == "no_iris_detected" else 422
        raise HTTPException(status_code=status, detail=exc.code) from exc

    return AnalysisResponse(
        quality=QualityInfo(
            blur_score=round(result.quality.blur_score, 2),
            overexposed_ratio=round(result.quality.overexposed_ratio, 4),
            eye_open=result.quality.eye_open,
            sample_pixel_count=result.lab.sample_pixel_count,
            issues=[],
        ),
        lab=LabValues(
            L=round(result.lab.L, 2),
            a=round(result.lab.a, 2),
            b=round(result.lab.b, 2),
        ),
        iris_color=IrisColorInfo(
            code=result.iris_color.code,
            label=result.iris_color.label,
            confidence=result.iris_color.confidence,
            reason=result.iris_color.reason,
        ),
        grade=result.grade.grade,
        confidence=result.grade.confidence,
        detection_method=result.detection.method,
    )
