"""轻量验证虹膜定位。

覆盖三类样例：
- 清晰特写：验证旧 eye_closeup 路径不退化
- 论文截图：验证反光瞳孔时仍能圈住虹膜主体
- 微信实拍：验证 auto 能在低对比图中给出可人工微调的初始圆
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
IMAGE_DIR = ROOT.parent / "img"


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


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise AssertionError(f"cannot load image: {path}")
    return image


def _case_paths(names: list[str]) -> list[Path]:
    return [IMAGE_DIR / name for name in names if (IMAGE_DIR / name).exists()]


def _assert_basic_geometry(name: str, image: np.ndarray, result, *, loose: bool = False) -> None:
    h, w = image.shape[:2]
    min_dim = min(h, w)
    det = result.detection
    cx, cy = det.center
    inner_radius = float(det.inner_radius or det.pupil_radius or 0)
    outer_radius = float(det.outer_radius or det.radius or 0)
    valid_count = int(result.sampling.valid.sum())

    assert 0 <= cx < w and 0 <= cy < h, f"{name}: center out of bounds: {(cx, cy)}"
    assert outer_radius > inner_radius + 2, (
        f"{name}: invalid ring radii inner={inner_radius:.2f}, outer={outer_radius:.2f}"
    )
    min_outer = min_dim * (0.035 if loose else 0.06)
    max_outer = min_dim * (0.46 if loose else 0.36)
    assert min_outer <= outer_radius <= max_outer, (
        f"{name}: outer radius unreasonable: {outer_radius:.2f}, expected {min_outer:.2f}-{max_outer:.2f}"
    )
    assert valid_count >= (50 if loose else 300), f"{name}: not enough valid samples: {valid_count}"


def _validate_case(config: dict, path: Path, mode: str, *, loose: bool = False) -> dict:
    image = _load_image(path)
    result = run_analysis(image, config, CONFIG_PATH, skip_quality=True, detection_mode=mode)
    _assert_basic_geometry(path.name, image, result, loose=loose)
    det = result.detection
    return {
        "file": path.name,
        "mode": mode,
        "method": det.method,
        "center": det.center,
        "pupil_radius": round(float(det.pupil_radius or 0), 2),
        "inner_radius": round(float(det.inner_radius or 0), 2),
        "outer_radius": round(float(det.outer_radius or det.radius), 2),
        "valid_sample_count": int(result.sampling.valid.sum()),
        "selection_score": None if det.selection_score is None else round(det.selection_score, 3),
    }


def main() -> None:
    config = load_config(CONFIG_PATH)
    reports = []

    synthetic = _load_sample()
    synthetic_result = run_analysis(
        synthetic,
        config,
        CONFIG_PATH,
        skip_quality=True,
        detection_mode="eye_closeup",
    )
    _assert_basic_geometry("synthetic_or_debug_sample", synthetic, synthetic_result)
    reports.append(
        {
            "file": "synthetic_or_debug_sample",
            "mode": "eye_closeup",
            "method": synthetic_result.detection.method,
            "outer_radius": round(float(synthetic_result.detection.outer_radius or 0), 2),
            "valid_sample_count": int(synthetic_result.sampling.valid.sum()),
        }
    )

    clear_cases = _case_paths(["test3.jpg", "test4.jpg", "test5.jpg", "test7.jpg"])
    paper_cases = _case_paths([f"new-{idx}.png" for idx in range(14, 25)])
    phone_cases = _case_paths(["WechatIMG1026.jpg", "WechatIMG1027.jpg"])

    for path in clear_cases:
        reports.append(_validate_case(config, path, "eye_closeup"))
    for path in paper_cases:
        reports.append(_validate_case(config, path, "auto", loose=True))
    for path in phone_cases:
        reports.append(_validate_case(config, path, "auto", loose=True))

    assert clear_cases, "no clear test images found"
    assert paper_cases, "no paper screenshot images found"
    assert phone_cases, "no phone capture images found"

    for report in reports:
        print(report)


if __name__ == "__main__":
    main()
