"""MediaPipe Face Landmarker 单例（Tasks API）。"""

from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

_MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "assets"
    / "models"
    / "face_landmarker.task"
)

_landmarker: vision.FaceLandmarker | None = None


def get_face_landmarker() -> vision.FaceLandmarker:
    """懒加载 FaceLandmarker，避免重复初始化。"""
    global _landmarker
    if _landmarker is None:
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"缺少模型文件: {_MODEL_PATH}。"
                "请从 MediaPipe 官方下载 face_landmarker.task 放到 assets/models/。"
            )
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(_MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        _landmarker = vision.FaceLandmarker.create_from_options(options)
    return _landmarker


def detect_face_landmarks(rgb_image):
    """
    检测人脸 landmark。
    rgb_image: numpy RGB uint8 数组
    返回 landmark 列表，未检测到则 None。
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = get_face_landmarker().detect(mp_image)
    if not result.face_landmarks:
        return None
    return result.face_landmarks[0]
