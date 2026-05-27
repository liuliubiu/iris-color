"""API 响应数据模型（Pydantic）。"""

from typing import List, Optional

from pydantic import BaseModel, Field


class LabValues(BaseModel):
    """CIELAB 颜色值。"""

    L: float = Field(..., description="明度，越低越深")
    a: float = Field(..., description="红绿轴")
    b: float = Field(..., description="黄蓝轴")


class QualityInfo(BaseModel):
    """图像质量检测结果。"""

    blur_score: float = Field(..., description="拉普拉斯方差，越大越清晰")
    overexposed_ratio: float = Field(..., description="过曝像素占比")
    eye_open: bool = Field(..., description="是否检测到睁眼")
    sample_pixel_count: int = Field(0, description="虹膜采样有效像素数")
    issues: List[str] = Field(default_factory=list, description="质量问题列表")


class AnalysisResponse(BaseModel):
    """分析成功响应。"""

    success: bool = True
    quality: QualityInfo
    lab: LabValues
    grade: int = Field(..., ge=1, le=5, description="Pan 2017 风格 5 档，1 最浅 5 最深")
    confidence: float = Field(..., ge=0.0, le=1.0, description="分档置信度")
    message: str = "provisional thresholds, calibration required"


class ErrorResponse(BaseModel):
    """分析失败响应。"""

    success: bool = False
    error: str
    quality: Optional[QualityInfo] = None


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = "ok"
    service: str = "iris-vision"
