"""对照人工标注真值评估检测精度（中心/半径误差）。

用法（在 iris-vision 目录，先在网页 /label/ui 标注若干图）：
    python scripts/eval_against_truth.py

真值来自 labels/ground_truth.json（原图坐标）；对每张已标注图跑 run_analysis，
把检测结果换算回原图坐标后与真值比较。误差按虹膜真值半径归一化，越小越准。
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.pipeline import AnalysisError, load_config, run_analysis  # noqa: E402

IMG_ROOT = ROOT.parent / "img"
GT_PATH = ROOT / "labels" / "ground_truth.json"
CONFIG_PATH = ROOT / "config" / "grade_thresholds.yaml"


def _imread(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def main() -> None:
    if not GT_PATH.exists():
        print(f"未找到真值文件 {GT_PATH}\n请先在网页 /label/ui 标注瞳孔/虹膜后再评估。")
        return
    truth = json.loads(GT_PATH.read_text(encoding="utf-8"))
    if not truth:
        print("真值为空，请先在网页标注。")
        return

    config = load_config(CONFIG_PATH)
    rows = []
    print(f"{'image':<42} {'ctrErr%':>8} {'irisErr%':>9} {'pupErr%':>8}  method")
    print("-" * 92)

    for rel in sorted(truth.keys()):
        gt = truth[rel]
        img = _imread(IMG_ROOT / rel)
        if img is None:
            print(f"{rel:<42} decode_failed")
            continue
        try:
            r = run_analysis(img, config, CONFIG_PATH, skip_quality=True)
        except AnalysisError as exc:
            print(f"{rel:<42} FAIL {exc.code}")
            rows.append(None)
            continue

        det = r.detection
        tf = r.transform
        dcx, dcy = det.pupil_center or det.center
        ocx, ocy = tf.to_original_xy(float(dcx), float(dcy))
        det_iris_r = tf.to_original_len(float(det.outer_radius or det.radius or 0))
        det_pupil_r = tf.to_original_len(float(det.pupil_radius or 0))

        gi, gp = gt["iris"], gt["pupil"]
        iris_r = max(gi["r"], 1.0)
        ctr_err = np.hypot(ocx - gi["cx"], ocy - gi["cy"]) / iris_r * 100
        iris_err = abs(det_iris_r - gi["r"]) / iris_r * 100
        pup_err = abs(det_pupil_r - gp["r"]) / iris_r * 100
        rows.append((ctr_err, iris_err, pup_err))
        print(f"{rel:<42} {ctr_err:8.1f} {iris_err:9.1f} {pup_err:8.1f}  {det.method}")

    valid = [x for x in rows if x is not None]
    print("-" * 92)
    if valid:
        arr = np.array(valid)
        med = np.median(arr, axis=0)
        mean = np.mean(arr, axis=0)
        p90 = np.percentile(arr, 90, axis=0)
        print(f"{'median':<42} {med[0]:8.1f} {med[1]:9.1f} {med[2]:8.1f}")
        print(f"{'mean':<42} {mean[0]:8.1f} {mean[1]:9.1f} {mean[2]:8.1f}")
        print(f"{'p90':<42} {p90[0]:8.1f} {p90[1]:9.1f} {p90[2]:8.1f}")
        print(f"\n评估 {len(valid)}/{len(truth)} 张（误差均为相对虹膜真值半径的百分比，越小越准）")
    else:
        print("无有效评估结果。")


if __name__ == "__main__":
    main()
