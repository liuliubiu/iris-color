"""一次性回归：对 img/ 全量（含子文件夹）跑检测，导出定位/环带/取色叠加图与汇总 JSON。

用法（在 iris-vision 目录）：
    python scripts/validate_imgs.py [auto|precise|rough] [--tag baseline]
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.debug_viz import build_debug_images, build_debug_metrics  # noqa: E402
from app.services.pipeline import AnalysisError, load_config, run_analysis  # noqa: E402

IMG_DIR = ROOT.parent / "img"
OUT_DIR = ROOT / "debug_output" / "_validate"
CONFIG_PATH = ROOT / "config" / "grade_thresholds.yaml"

WANTED = ("01_pupil_localization", "02_iris_ring", "04_valid_samples")
EXTS = (".jpg", ".jpeg", ".png")


def _collect_images() -> list:
    paths = [p for p in IMG_DIR.rglob("*") if p.suffix.lower() in EXTS and p.is_file()]
    return sorted(paths, key=lambda p: str(p.relative_to(IMG_DIR)))


def _stem(path: Path) -> str:
    rel = path.relative_to(IMG_DIR)
    parts = list(rel.parts[:-1]) + [rel.stem]
    return "__".join(parts)


def _imread_unicode(path: Path):
    """cv2.imread 在 Windows 上读不了中文路径，改走字节流解码。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _imwrite_unicode(path: Path, image) -> None:
    ok, buf = cv2.imencode(path.suffix, image)
    if ok:
        buf.tofile(str(path))


def main() -> None:
    args = [a for a in sys.argv[1:]]
    tag = "run"
    if "--tag" in args:
        i = args.index("--tag")
        tag = args[i + 1]
        del args[i : i + 2]
    mode = args[0] if args else "auto"

    config = load_config(CONFIG_PATH)
    highlight_v = config.get("highlight_v_threshold", 240)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    images = _collect_images()
    print(f"mode={mode}  tag={tag}  images={len(images)}  out={OUT_DIR}")
    print("-" * 100)

    summary = {}
    for path in images:
        name = _stem(path)
        image_bgr = _imread_unicode(path)
        if image_bgr is None:
            print(f"{name:<44} SKIP (decode failed)")
            summary[name] = {"status": "decode_failed"}
            continue
        t0 = time.perf_counter()
        try:
            pipeline = run_analysis(
                image_bgr, config, CONFIG_PATH, skip_quality=True, closeup_mode=mode
            )
        except AnalysisError as exc:
            elapsed = time.perf_counter() - t0
            print(f"{name:<44} FAIL  {exc.code}  ({elapsed:.2f}s)")
            summary[name] = {"status": "fail", "error": exc.code, "elapsed_s": round(elapsed, 3)}
            continue
        elapsed = time.perf_counter() - t0

        metrics = build_debug_metrics(pipeline, highlight_v, config=config)
        overlays = build_debug_images(
            pipeline,
            config.get("eye_closeup", {}),
            config=config,
            highlight_v=highlight_v,
        )
        for key in WANTED:
            _imwrite_unicode(OUT_DIR / f"{name}__{key}.jpg", overlays[key])

        det = pipeline.detection
        summary[name] = {
            "status": "ok",
            "elapsed_s": round(elapsed, 3),
            "method": det.method,
            "pupil_radius": round(float(det.pupil_radius or 0), 1),
            "outer_radius": round(float(det.outer_radius or det.radius), 1),
            "valid": metrics["valid_sample_count"],
            "lab": metrics["lab"],
            "grade": metrics["grade"],
            "color": metrics["iris_color"]["code"],
        }
        print(
            f"{name:<44} {elapsed:5.2f}s method={det.method:<20} "
            f"pupil_r={det.pupil_radius:7.1f} outer_r={det.outer_radius:7.1f} "
            f"valid={metrics['valid_sample_count']:<7} "
            f"L={metrics['lab']['L']:6.2f} grade={metrics['grade']} "
            f"color={metrics['iris_color']['code']}"
        )

    out_json = OUT_DIR / f"summary_{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    ok = sum(1 for v in summary.values() if v.get("status") == "ok")
    total_time = sum(v.get("elapsed_s", 0) for v in summary.values())
    print("-" * 100)
    print(f"ok={ok}/{len(summary)}  total={total_time:.1f}s  summary={out_json}")


if __name__ == "__main__":
    main()
