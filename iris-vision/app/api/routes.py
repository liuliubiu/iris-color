"""FastAPI 路由定义。"""

import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models.schemas import (
    AnalysisResponse,
    DetectionInfo,
    ErrorResponse,
    HealthResponse,
    IrisColorInfo,
    LabValues,
    QualityInfo,
)
from app.services.debug_viz import build_debug_images, images_to_base64
from app.services.pipeline import AnalysisError, load_config, run_analysis

router = APIRouter()

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "grade_thresholds.yaml"


def _decode_image(content: bytes) -> np.ndarray:
    arr = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="invalid_image_format")
    return image


def _parse_manual_params(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="invalid_manual_params_json") from exc
    required = ["center_x", "center_y", "pupil_radius", "inner_radius", "outer_radius"]
    if not isinstance(data, dict) or any(key not in data for key in required):
        raise HTTPException(status_code=400, detail="manual_params_missing_fields")
    try:
        return {key: float(data[key]) for key in required}
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="manual_params_must_be_numbers") from exc


def _build_result_debug_images(image_bgr: np.ndarray, result, config: dict) -> dict:
    images = build_debug_images(image_bgr, result, config.get("eye_closeup", {}))
    wanted = {
        "01_pupil_localization": images["01_pupil_localization"],
        "02_iris_ring": images["02_iris_ring"],
        "04_valid_samples": images["04_valid_samples"],
    }
    return images_to_base64(wanted)


def _to_analysis_response(result, image_bgr: np.ndarray, config: dict) -> AnalysisResponse:
    det = result.detection
    center_x, center_y = det.pupil_center or det.center
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
        detection=DetectionInfo(
            center_x=int(center_x),
            center_y=int(center_y),
            pupil_radius=float(det.pupil_radius or 0),
            inner_radius=float(det.inner_radius or 0),
            outer_radius=float(det.outer_radius or det.radius),
            method=det.method,
        ),
        debug_images=_build_result_debug_images(image_bgr, result, config),
        grade=result.grade.grade,
        confidence=result.grade.confidence,
        detection_method=result.detection.method,
    )


def _raise_analysis_error(exc: AnalysisError) -> None:
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
        _raise_analysis_error(exc)

    return _to_analysis_response(result, image_bgr, config)


@router.post(
    "/analyze/manual",
    response_model=AnalysisResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def analyze_manual(
    file: UploadFile = File(...),
    manual_params: str = Form(..., description="JSON: center_x, center_y, pupil_radius, inner_radius, outer_radius"),
) -> AnalysisResponse:
    """上传眼部特写和人工调整参数，重新返回 CIELAB、颜色与 Grade。"""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file_must_be_image")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty_file")

    config = load_config(CONFIG_PATH)
    image_bgr = _decode_image(content)
    manual_detection = _parse_manual_params(manual_params)

    try:
        result = run_analysis(image_bgr, config, CONFIG_PATH, manual_detection=manual_detection)
    except AnalysisError as exc:
        _raise_analysis_error(exc)

    return _to_analysis_response(result, image_bgr, config)
