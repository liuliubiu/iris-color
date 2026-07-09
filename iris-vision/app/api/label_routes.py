"""人工标注后门：在网页上标注镜筒图的正确瞳孔/虹膜圆，保存为真值。

仅供开发标定使用（复用 debug api_key）。真值保存到 iris-vision/labels/ground_truth.json，
坐标为原图像素。评估脚本 scripts/eval_against_truth.py 据此量化检测误差。
"""

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import HTMLResponse, Response

from app.services.pipeline import AnalysisError, load_config, run_analysis

router = APIRouter(prefix="/label", tags=["label"])

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "config" / "grade_thresholds.yaml"
IMG_ROOT = ROOT.parent / "img"
GT_PATH = ROOT / "labels" / "ground_truth.json"
EXTS = (".jpg", ".jpeg", ".png")
_DISPLAY_MAX = 1000  # 传给浏览器的最长边（原图坐标由前端按比例还原）


def _verify_key(key: Optional[str]) -> dict:
    config = load_config(CONFIG_PATH)
    debug_cfg = config.get("debug", {})
    if not debug_cfg.get("enabled", True):
        raise HTTPException(status_code=403, detail="debug_disabled")
    if key != debug_cfg.get("api_key", "iris-color-dev"):
        raise HTTPException(status_code=403, detail="invalid_debug_key")
    return config


def _imread_unicode(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _safe_rel(rel: str) -> Path:
    """把前端传来的相对路径限制在 img/ 目录内，避免路径穿越。"""
    rel = (rel or "").replace("\\", "/").lstrip("/")
    target = (IMG_ROOT / rel).resolve()
    if not str(target).startswith(str(IMG_ROOT.resolve())):
        raise HTTPException(status_code=400, detail="invalid_path")
    return target


def _list_images() -> list:
    """列出 img/ 子文件夹内的镜筒实拍图（跳过顶层零散测试图）。"""
    if not IMG_ROOT.exists():
        return []
    items = []
    for path in IMG_ROOT.rglob("*"):
        if path.suffix.lower() not in EXTS or not path.is_file():
            continue
        if path.parent == IMG_ROOT:
            continue  # 只标注子文件夹内的实拍图
        rel = path.relative_to(IMG_ROOT).as_posix()
        items.append(rel)
    return sorted(items)


def _load_truth() -> dict:
    if GT_PATH.exists():
        try:
            return json.loads(GT_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_truth(data: dict) -> None:
    GT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GT_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


@router.get("/list")
def list_images(key: str = Query(...)):
    _verify_key(key)
    truth = _load_truth()
    out = []
    for rel in _list_images():
        img = _imread_unicode(IMG_ROOT / rel)
        if img is None:
            continue
        h, w = img.shape[:2]
        out.append(
            {
                "rel": rel,
                "width": int(w),
                "height": int(h),
                "labeled": rel in truth,
            }
        )
    return {"images": out, "count": len(out), "labeled": len(truth)}


@router.get("/image")
def get_image(rel: str = Query(...), key: str = Query(...)):
    _verify_key(key)
    path = _safe_rel(rel)
    img = _imread_unicode(path)
    if img is None:
        raise HTTPException(status_code=404, detail="image_not_found")
    h, w = img.shape[:2]
    scale = _DISPLAY_MAX / max(h, w) if max(h, w) > _DISPLAY_MAX else 1.0
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise HTTPException(status_code=500, detail="encode_failed")
    return Response(
        content=buf.tobytes(),
        media_type="image/jpeg",
        headers={"X-Orig-Width": str(w), "X-Orig-Height": str(h)},
    )


@router.get("/auto")
def auto_detect(rel: str = Query(...), key: str = Query(...)):
    """跑当前检测，返回原图坐标下的瞳孔/虹膜圆，供标注预填。"""
    config = _verify_key(key)
    path = _safe_rel(rel)
    img = _imread_unicode(path)
    if img is None:
        raise HTTPException(status_code=404, detail="image_not_found")
    try:
        result = run_analysis(img, config, CONFIG_PATH, skip_quality=True)
    except AnalysisError as exc:
        return {"ok": False, "error": exc.code}
    det = result.detection
    tf = result.transform
    cx, cy = det.pupil_center or det.center
    ocx, ocy = tf.to_original_xy(float(cx), float(cy))
    return {
        "ok": True,
        "pupil": {
            "cx": round(ocx, 1),
            "cy": round(ocy, 1),
            "r": round(tf.to_original_len(float(det.pupil_radius or 0)), 1),
        },
        "iris": {
            "cx": round(ocx, 1),
            "cy": round(ocy, 1),
            "r": round(tf.to_original_len(float(det.outer_radius or det.radius or 0)), 1),
        },
        "method": det.method,
    }


@router.get("/truth")
def get_truth(key: str = Query(...)):
    _verify_key(key)
    return _load_truth()


@router.post("/truth")
def save_truth(key: str = Query(...), payload: dict = Body(...)):
    _verify_key(key)
    rel = payload.get("rel")
    if not rel:
        raise HTTPException(status_code=400, detail="missing_rel")
    _safe_rel(rel)  # 校验路径合法
    try:
        pupil = payload["pupil"]
        iris = payload["iris"]
        entry = {
            "pupil": {"cx": float(pupil["cx"]), "cy": float(pupil["cy"]), "r": float(pupil["r"])},
            "iris": {"cx": float(iris["cx"]), "cy": float(iris["cy"]), "r": float(iris["r"])},
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="invalid_payload") from exc

    data = _load_truth()
    data[rel] = entry
    _save_truth(data)
    return {"ok": True, "rel": rel, "total": len(data)}


@router.delete("/truth")
def delete_truth(rel: str = Query(...), key: str = Query(...)):
    _verify_key(key)
    data = _load_truth()
    if rel in data:
        del data[rel]
        _save_truth(data)
    return {"ok": True, "rel": rel, "total": len(data)}


@router.get("/ui", response_class=HTMLResponse)
def label_ui():
    config = load_config(CONFIG_PATH)
    if not config.get("debug", {}).get("enabled", True):
        raise HTTPException(status_code=403, detail="debug_disabled")
    html_path = ROOT / "app" / "static" / "label_ui.html"
    if not html_path.exists():
        raise HTTPException(status_code=500, detail="label_ui_missing")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))
