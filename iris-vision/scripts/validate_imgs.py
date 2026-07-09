"""一次性回归：对 img/（含人名子目录）全量跑检测，导出定位/环带/取色叠加图。

用法（在 iris-vision 目录）：
    python scripts/validate_imgs.py [auto|precise|rough]
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.debug_viz import build_debug_images, build_debug_metrics  # noqa: E402
from app.services.eye_iris_detect import crop_circular_fov  # noqa: E402
from app.services.pipeline import AnalysisError, load_config, run_analysis  # noqa: E402
from app.services.quality import compute_blur_score  # noqa: E402

IMG_DIR = ROOT.parent / "img"
OUT_DIR = ROOT / "debug_output" / "_validate"
CONFIG_PATH = ROOT / "config" / "grade_thresholds.yaml"

WANTED = ("01_pupil_localization", "02_iris_ring", "04_valid_samples")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _collect_images(root: Path) -> list[Path]:
    """递归收集 img/ 下图片（含霍/李/林等人名子目录）。"""
    if not root.is_dir():
        return []
    images = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(images, key=lambda p: str(p.relative_to(root)).lower())


def _imread_unicode(path: Path):
    """Windows 下 OpenCV imread 无法处理非 ASCII 路径，改用 imdecode。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "auto"
    config = load_config(CONFIG_PATH)
    highlight_v = config.get("highlight_v_threshold", 240)
    eye_cfg = config.get("eye_closeup", {})
    fov_cfg = eye_cfg.get("circular_fov", {})
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    images = _collect_images(IMG_DIR)
    print(f"mode={mode}  images={len(images)}  out={OUT_DIR}")
    print("-" * 110)

    ok = 0
    fail = 0
    total_ms = 0.0

    for path in images:
        rel = path.relative_to(IMG_DIR)
        image_bgr = _imread_unicode(path)
        if image_bgr is None:
            print(f"{str(rel):<42} SKIP (decode failed)")
            fail += 1
            continue

        fov_applied = False
        if fov_cfg.get("enabled", True):
            fov = crop_circular_fov(
                image_bgr,
                luminance_threshold=int(fov_cfg.get("luminance_threshold", 12)),
                min_coverage=float(fov_cfg.get("min_coverage", 0.15)),
                max_coverage=float(fov_cfg.get("max_coverage", 0.92)),
                margin_ratio=float(fov_cfg.get("margin_ratio", 0.02)),
            )
            fov_applied = fov.applied

        t0 = time.perf_counter()
        try:
            pipeline = run_analysis(
                image_bgr, config, CONFIG_PATH, skip_quality=True, closeup_mode=mode
            )
        except AnalysisError as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            total_ms += elapsed_ms
            fail += 1
            print(
                f"{str(rel):<42} FAIL  {exc.code:<32} "
                f"fov={int(fov_applied)}  {elapsed_ms:6.0f}ms"
            )
            continue

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_ms += elapsed_ms
        ok += 1

        metrics = build_debug_metrics(pipeline, highlight_v)
        overlays = build_debug_images(image_bgr, pipeline, eye_cfg)
        # 子目录文件名可能冲突，用相对路径扁平化
        stem = str(rel).replace("\\", "__").replace("/", "__")
        stem = Path(stem).stem
        for name in WANTED:
            cv2.imwrite(str(OUT_DIR / f"{stem}__{name}.jpg"), overlays[name])

        det = pipeline.detection
        blur = compute_blur_score(image_bgr)
        print(
            f"{str(rel):<42} method={det.method:<20} "
            f"pupil_r={det.pupil_radius:.1f} outer_r={det.outer_radius:.1f} "
            f"valid={metrics['valid_sample_count']:<6} "
            f"pconf={det.pupil_confidence:.2f} iconf={det.iris_confidence:.2f} "
            f"color={metrics['iris_color']['code']:<12} "
            f"blur={blur:6.1f} fov={int(fov_applied)}  {elapsed_ms:6.0f}ms"
        )

    print("-" * 110)
    avg = total_ms / max(ok + fail, 1)
    print(
        f"done  ok={ok}  fail={fail}  "
        f"total={total_ms:.0f}ms  avg={avg:.0f}ms"
    )


if __name__ == "__main__":
    main()
