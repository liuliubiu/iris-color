"""一次性回归：对指定 img/ 目录跑检测，导出定位/环带/取色叠加图。

用法（在 iris-vision 目录）：
    python scripts/validate_imgs.py [auto|precise|rough] [目录名...]

默认验证当前实拍基准目录：霍、李、林、刘、罗、其它实拍图、吴、杨。
"""

import json
import sys
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
DEFAULT_DIRS = ("霍", "李", "林", "刘", "罗", "其它实拍图", "吴", "杨")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def _parse_args(argv: list[str]) -> tuple[str, list[str]]:
    mode = "auto"
    dirs: list[str] = []
    if argv and argv[0] in ("auto", "precise", "rough"):
        mode = argv[0]
        dirs = argv[1:]
    else:
        dirs = argv
    return mode, dirs or list(DEFAULT_DIRS)


def _iter_images(dir_names: list[str]) -> list[Path]:
    images: list[Path] = []
    for name in dir_names:
        path = IMG_DIR / name
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
            continue
        if not path.exists():
            print(f"WARN missing: {path}")
            continue
        images.extend(
            p
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
    return sorted(images, key=lambda p: p.relative_to(IMG_DIR).as_posix())


def _output_stem(path: Path) -> str:
    rel = path.relative_to(IMG_DIR).with_suffix("")
    return "__".join(rel.parts)


def _read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _write_image(path: Path, image) -> bool:
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def main() -> None:
    mode, dir_names = _parse_args(sys.argv[1:])
    config = load_config(CONFIG_PATH)
    highlight_v = config.get("highlight_v_threshold", 240)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    images = _iter_images(dir_names)
    summary = {
        "mode": mode,
        "dirs": dir_names,
        "images": {},
    }
    print(f"mode={mode}  dirs={','.join(dir_names)}  images={len(images)}  out={OUT_DIR}")
    print("-" * 92)

    for path in images:
        rel_name = path.relative_to(IMG_DIR).as_posix()
        image_bgr = _read_image(path)
        if image_bgr is None:
            summary["images"][rel_name] = {"status": "decode_failed"}
            print(f"{rel_name:<42} SKIP (decode failed)")
            continue
        try:
            pipeline = run_analysis(
                image_bgr, config, CONFIG_PATH, skip_quality=True, closeup_mode=mode
            )
        except AnalysisError as exc:
            summary["images"][rel_name] = {"status": "fail", "code": exc.code}
            print(f"{rel_name:<42} FAIL  {exc.code}")
            continue

        metrics = build_debug_metrics(pipeline, highlight_v)
        overlays = build_debug_images(image_bgr, pipeline, config.get("eye_closeup", {}))
        stem = _output_stem(path)
        for name in WANTED:
            _write_image(OUT_DIR / f"{stem}__{name}.jpg", overlays[name])

        det = pipeline.detection
        summary["images"][rel_name] = {
            "status": "ok",
            "method": det.method,
            "center": list(det.center),
            "pupil_center": list(det.pupil_center) if det.pupil_center else None,
            "pupil_radius": det.pupil_radius,
            "inner_radius": det.inner_radius,
            "outer_radius": det.outer_radius,
            "pupil_confidence": det.pupil_confidence,
            "iris_confidence": det.iris_confidence,
            "valid_sample_count": metrics["valid_sample_count"],
            "lab": metrics["lab"],
            "grade": metrics["grade"],
            "color": metrics["iris_color"]["code"],
        }
        print(
            f"{rel_name:<42} method={det.method:<20} "
            f"pupil_r={det.pupil_radius:.1f} outer_r={det.outer_radius:.1f} "
            f"valid={metrics['valid_sample_count']:<6} "
            f"pconf={det.pupil_confidence:.2f} iconf={det.iris_confidence:.2f} "
            f"color={metrics['iris_color']['code']}"
        )

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
