import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.services.scope_field import detect_scope_field  # noqa: E402

truth = json.loads((ROOT / "labels" / "ground_truth.json").read_text(encoding="utf-8"))
IMG = ROOT.parent / "img"


def rd(p):
    d = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(d, cv2.IMREAD_COLOR)


ratios, pratios, offs, poffs = [], [], [], []
for rel, gt in sorted(truth.items()):
    img = rd(IMG / rel)
    sc = detect_scope_field(img)
    if sc is None:
        print("noscope", rel)
        continue
    ir = gt["iris"]["r"] / sc.radius
    pr = gt["pupil"]["r"] / sc.radius
    off = np.hypot(gt["iris"]["cx"] - sc.center_x, gt["iris"]["cy"] - sc.center_y) / sc.radius
    poff = np.hypot(gt["pupil"]["cx"] - gt["iris"]["cx"], gt["pupil"]["cy"] - gt["iris"]["cy"]) / gt["iris"]["r"]
    ratios.append(ir); pratios.append(pr); offs.append(off); poffs.append(poff)
    print(f"{rel[:24]:26} iris/scope={ir:.3f} pup/scope={pr:.3f} irisCoff={off:.3f} pup-iris/irisR={poff:.3f}")

print("--- iris/scope : min %.3f  med %.3f  max %.3f" % (min(ratios), np.median(ratios), max(ratios)))
print("--- pup/scope  : min %.3f  med %.3f  max %.3f" % (min(pratios), np.median(pratios), max(pratios)))
print("--- irisCoff   : med %.3f  max %.3f" % (np.median(offs), max(offs)))
print("--- pupVsIris  : med %.3f  max %.3f" % (np.median(poffs), max(poffs)))
