"""临时诊断：把检测结果(青/黄) 与人工真值(绿=虹膜, 蓝=瞳孔) 叠加对比。"""

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
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "iris_gt"


def _imread(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


config = load_config(CONFIG_PATH)
truth = json.loads(GT_PATH.read_text(encoding="utf-8"))
OUT.mkdir(parents=True, exist_ok=True)

idx = 0
for rel in sorted(truth.keys()):
    idx += 1
    gt = truth[rel]
    img = _imread(IMG_ROOT / rel)
    if img is None:
        continue
    canvas = img.copy()
    gi, gp = gt["iris"], gt["pupil"]
    # 真值：绿=虹膜 蓝=瞳孔
    cv2.circle(canvas, (int(gi["cx"]), int(gi["cy"])), int(gi["r"]), (0, 255, 0), 6)
    cv2.circle(canvas, (int(gp["cx"]), int(gp["cy"])), int(gp["r"]), (255, 60, 0), 6)
    det_txt = "FAIL"
    try:
        r = run_analysis(img, config, CONFIG_PATH, skip_quality=True)
        det = r.detection
        tf = r.transform
        dcx, dcy = det.pupil_center or det.center
        ocx, ocy = tf.to_original_xy(float(dcx), float(dcy))
        det_iris = tf.to_original_len(float(det.outer_radius or det.radius or 0))
        det_pup = tf.to_original_len(float(det.pupil_radius or 0))
        # 检测：青=虹膜外缘 黄=瞳孔
        cv2.circle(canvas, (int(ocx), int(ocy)), int(det_iris), (255, 255, 0), 5)
        cv2.circle(canvas, (int(ocx), int(ocy)), int(det_pup), (0, 255, 255), 5)
        ctr = np.hypot(ocx - gi["cx"], ocy - gi["cy"]) / max(gi["r"], 1) * 100
        det_txt = f"ctr{ctr:.0f}% irisD{det_iris:.0f} irisGT{gi['r']:.0f} {det.method}"
    except AnalysisError as exc:
        det_txt = f"FAIL {exc.code}"
    scale = 560.0 / max(canvas.shape[:2])
    canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.putText(canvas, f"{idx:02d} {det_txt}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    buf.tofile(str(OUT / f"gt_{idx:02d}.jpg"))
    print(f"{idx:02d} {rel}  {det_txt}")
