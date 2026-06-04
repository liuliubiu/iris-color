"""API 响应数据模型（Pydantic）。"""

from typing import List, Optional

from pydantic import BaseModel, Field


class LabValues(BaseModel):
    """CIELAB 颜色值。"""

    L: float = Field(..., description="明度，越低越深")
    a: float = Field(..., description="红绿轴")
    b: float = Field(..., description="黄蓝轴")


class IrisColorInfo(BaseModel):
    """基础虹膜颜色判断。"""

    code: str = Field(..., description="颜色分类代码")
    label: str = Field(..., description="中文颜色名称")
    confidence: float = Field(..., ge=0.0, le=1.0, description="颜色判断置信度")
    reason: str = Field(..., description="主要判定依据")


class DetectionInfo(BaseModel):
    """前端人工调整所需的定位参数。"""

    center_x: int = Field(..., description="瞳孔/虹膜环带中心 x")
    center_y: int = Field(..., description="瞳孔/虹膜环带中心 y")
    pupil_radius: float = Field(..., description="瞳孔半径")
    inner_radius: float = Field(..., description="取色环带内半径")
    outer_radius: float = Field(..., description="取色环带外半径")
    method: str = Field(..., description="定位方式")


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
    iris_color: IrisColorInfo
    detection: DetectionInfo
    debug_images: dict = Field(default_factory=dict, description="前端展示识别依据用的调试图 base64")
    grade: int = Field(..., ge=1, le=5, description="Pan 2017 风格 5 档，1 最浅 5 最深")
    confidence: float = Field(..., ge=0.0, le=1.0, description="分档置信度")
    detection_method: str = Field(
        "eye_closeup",
        description="虹膜定位方式：eye_closeup、rough_closeup、face_landmark 或 manual_adjustment",
    )
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


class DebugAnalysisResponse(BaseModel):
    """调试分析响应（仅开发/标定使用）。"""

    success: bool = True
    run_id: Optional[str] = None
    saved_dir: Optional[str] = None
    viewer_url: Optional[str] = None
    metrics: dict = Field(default_factory=dict)
    image_urls: dict = Field(default_factory=dict, description="各阶段图片相对 URL")
    images_base64: Optional[dict] = Field(None, description="include_base64=true 时返回")
    message: str = ""
