"""一次性回归：对 img/ 全量跑检测，导出定位/环带/取色叠加图。

用法（在 iris-vision 目录）：
    python scripts/validate_imgs.py [auto|precise|rough]
"""

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.debug_viz import build_debug_images, build_debug_metrics  # noqa: E402
from app.services.pipeline import AnalysisError, load_config, run_analysis  # noqa: E402

IMG_DIR = ROOT.parent / "img"
OUT_DIR = ROOT / "debug_output" / "_validate"
CONFIG_PATH = ROOT / "config" / "grade_thresholds.yaml"

WANTED = ("01_pupil_localization", "02_iris_ring", "04_valid_samples")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "auto"
    config = load_config(CONFIG_PATH)
    highlight_v = config.get("highlight_v_threshold", 240)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print(f"mode={mode}  images={len(images)}  out={OUT_DIR}")
    print("-" * 92)

    for path in images:
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            print(f"{path.name:<22} SKIP (decode failed)")
            continue
        try:
            pipeline = run_analysis(
                image_bgr, config, CONFIG_PATH, skip_quality=True, closeup_mode=mode
            )
        except AnalysisError as exc:
            print(f"{path.name:<22} FAIL  {exc.code}")
            continue

        metrics = build_debug_metrics(pipeline, highlight_v)
        overlays = build_debug_images(image_bgr, pipeline, config.get("eye_closeup", {}))
        stem = path.stem
        for name in WANTED:
            cv2.imwrite(str(OUT_DIR / f"{stem}__{name}.jpg"), overlays[name])

        det = pipeline.detection
        print(
            f"{path.name:<22} method={det.method:<20} "
            f"pupil_r={det.pupil_radius:.1f} outer_r={det.outer_radius:.1f} "
            f"valid={metrics['valid_sample_count']:<6} "
            f"pconf={det.pupil_confidence:.2f} iconf={det.iris_confidence:.2f} "
            f"color={metrics['iris_color']['code']}"
        )


if __name__ == "__main__":
    main()
