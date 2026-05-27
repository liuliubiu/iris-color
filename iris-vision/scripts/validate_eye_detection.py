"""轻量验证眼部特写瞳孔/虹膜定位。

默认优先使用 debug_output 中的 synthetic 样例；不存在时现场生成一个简单样例。
"""

from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.pipeline import load_config, run_analysis


CONFIG_PATH = ROOT / "config" / "grade_thresholds.yaml"
DEFAULT_SAMPLE = ROOT / "debug_output" / "20260527_102923" / "00_original.jpg"


def _build_synthetic_eye() -> np.ndarray:
    image = np.full((400, 400, 3), (130, 145, 170), dtype=np.uint8)
    cv2.circle(image, (200, 200), 118, (95, 120, 150), -1)
    cv2.circle(image, (200, 200), 40, (0, 0, 0), -1)
    return image


def _load_sample() -> np.ndarray:
    if DEFAULT_SAMPLE.exists():
        sample = cv2.imread(str(DEFAULT_SAMPLE))
        if sample is not None:
            return sample
    return _build_synthetic_eye()


def main() -> None:
    config = load_config(CONFIG_PATH)
    image = _load_sample()
    result = run_analysis(image, config, CONFIG_PATH, skip_quality=True)
    pupil_radius = result.detection.pupil_radius or 0
    outer_radius = result.detection.outer_radius or 0

    assert 25 <= pupil_radius <= 60, f"pupil radius out of range: {pupil_radius:.2f}"
    assert outer_radius > pupil_radius * 2.0, (
        f"iris ring too close to pupil: pupil={pupil_radius:.2f}, outer={outer_radius:.2f}"
    )
    assert result.sampling.valid.sum() > 1000, "not enough valid iris samples"

    print(
        {
            "pupil_radius": round(pupil_radius, 2),
            "outer_radius": round(outer_radius, 2),
            "pupil_method": result.detection.pupil_method,
            "iris_outer_method": result.detection.iris_outer_method,
            "valid_sample_count": int(result.sampling.valid.sum()),
        }
    )


if __name__ == "__main__":
    main()
