"""下载 MediaPipe Face Landmarker 模型（首次运行前执行一次）。"""

from pathlib import Path
import urllib.request

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_DIR = Path(__file__).resolve().parent.parent / "assets" / "models"
MODEL_PATH = MODEL_DIR / "face_landmarker.task"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        print(f"模型已存在: {MODEL_PATH}")
        return
    print(f"正在下载模型到 {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("下载完成。")


if __name__ == "__main__":
    main()
