"""MediaPipe 眼部/虹膜先验：给出眼部 ROI 与虹膜中心/半径粗估。

复用全脸 FaceLandmarker（含 478 虹膜 landmark）。当模型缺失、依赖异常或图中
检不到人脸时一律返回 None，调用方据此回退到纯几何启发式，保证不破坏现有流程。
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.services.face_landmarker import detect_face_landmarks

_LEFT_IRIS_CENTER = 468
_RIGHT_IRIS_CENTER = 473
_IRIS_EDGES = {
    _LEFT_IRIS_CENTER: [469, 470, 471, 472],
    _RIGHT_IRIS_CENTER: [474, 475, 476, 477],
}


@dataclass
class EyePrior:
    """眼部先验：眼部 ROI 矩形 + 虹膜中心/半径粗估（均为输入图像素坐标）。"""

    roi: Tuple[int, int, int, int]  # x, y, w, h
    iris_cx: float
    iris_cy: float
    iris_r: float
    eye_side: str


def _iris_radius(landmarks, center_idx: int, w: int, h: int) -> float:
    center = landmarks[center_idx]
    cx, cy = center.x * w, center.y * h
    dists: List[float] = []
    for idx in _IRIS_EDGES.get(center_idx, []):
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        dx = lm.x * w - cx
        dy = lm.y * h - cy
        dists.append((dx * dx + dy * dy) ** 0.5)
    return float(np.mean(dists)) if dists else 0.0


def detect_eye_prior(
    image_bgr: np.ndarray,
    roi_margin: float = 3.0,
) -> Optional[EyePrior]:
    """
    在图像上跑 FaceLandmarker，取离画面中心最近的眼睛，返回虹膜中心/半径与眼部 ROI。

    roi_margin: ROI 半边长相对虹膜半径的倍数（越大 ROI 越宽松）。
    任何失败（缺模型/无脸/异常）都返回 None。
    """
    try:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        landmarks = detect_face_landmarks(rgb)
    except Exception:
        return None
    if landmarks is None:
        return None

    h, w = image_bgr.shape[:2]
    img_cx, img_cy = w / 2.0, h / 2.0

    best: Optional[Tuple[float, float, float, float, str]] = None
    best_dist = float("inf")
    for center_idx, side in [(_LEFT_IRIS_CENTER, "left"), (_RIGHT_IRIS_CENTER, "right")]:
        if center_idx >= len(landmarks):
            continue
        c = landmarks[center_idx]
        cx, cy = c.x * w, c.y * h
        r = _iris_radius(landmarks, center_idx, w, h)
        if r < 3.0:
            continue
        dist = (cx - img_cx) ** 2 + (cy - img_cy) ** 2
        if dist < best_dist:
            best_dist = dist
            best = (cx, cy, r, dist, side)

    if best is None:
        return None

    cx, cy, r, _, side = best
    half = max(r * roi_margin, r + 8.0)
    x1 = int(max(cx - half, 0))
    y1 = int(max(cy - half, 0))
    x2 = int(min(cx + half, w))
    y2 = int(min(cy + half, h))
    roi_w = x2 - x1
    roi_h = y2 - y1
    if roi_w < 8 or roi_h < 8:
        return None

    return EyePrior(
        roi=(x1, y1, roi_w, roi_h),
        iris_cx=float(cx),
        iris_cy=float(cy),
        iris_r=float(r),
        eye_side=side,
    )
