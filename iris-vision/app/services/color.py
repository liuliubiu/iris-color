"""RGB 转 CIELAB，计算虹膜区域颜色中位数。"""

from dataclasses import dataclass
from typing import Tuple

import cv2
import colorspacious
import numpy as np


@dataclass
class LabResult:
    """Lab 中位数结果。"""

    L: float
    a: float
    b: float
    sample_pixel_count: int


def extract_iris_lab_median(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    highlight_v_threshold: int = 240,
) -> LabResult:
    """
    在 mask 区域内取色，剔除高光像素，返回 CIELAB 中位数。

    步骤：
    1. 用 mask 选出虹膜环带像素
    2. HSV 中 V > threshold 的像素视为高光，排除
    3. RGB [0,255] → CIELAB（colorspacious，D65）
    4. 对 L*, a*, b* 分别取中位数（抗异常值）
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]

    valid = (mask > 0) & (v_channel < highlight_v_threshold)
    pixels_bgr = image_bgr[valid]

    if len(pixels_bgr) == 0:
        raise ValueError("no_valid_pixels_after_highlight_removal")

    # BGR → RGB，归一化到 [0, 1]
    pixels_rgb = pixels_bgr[:, ::-1].astype(np.float64) / 255.0

    # colorspacious: sRGB → CIELAB
    lab_array = colorspacious.cspace_convert(pixels_rgb, "sRGB1", "CIELab")

    l_median = float(np.median(lab_array[:, 0]))
    a_median = float(np.median(lab_array[:, 1]))
    b_median = float(np.median(lab_array[:, 2]))

    return LabResult(
        L=l_median,
        a=a_median,
        b=b_median,
        sample_pixel_count=int(len(pixels_bgr)),
    )
