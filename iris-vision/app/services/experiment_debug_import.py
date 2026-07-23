"""从 debug 运行结果映射到实验记录字段。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from app.services.experiment_store import COLOR_VALUES

_COMPARE_PANEL_W = 420
_COMPARE_GAP_W = 12
_COMPARE_FOOTER_H = 56

# classify_iris_color 中文标签 → 实验记录 9 色
IRIS_LABEL_TO_COLOR = {
    "浅蓝色": "浅蓝",
    "蓝色": "蓝",
    "深蓝色": "深蓝",
    "浅绿色": "浅绿",
    "绿色": "绿",
    "深绿色": "深绿",
    "浅棕色": "浅棕",
    "棕色": "棕",
    "深棕色": "深棕",
}


def grade_int_to_str(grade: Optional[int]) -> Optional[str]:
    if grade is None:
        return None
    try:
        g = int(grade)
    except (TypeError, ValueError):
        return None
    if 1 <= g <= 5:
        return f"Grade{g}"
    return None


def map_iris_color_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    mapped = IRIS_LABEL_TO_COLOR.get(label.strip())
    if mapped and mapped in COLOR_VALUES:
        return mapped
    # 模糊匹配：含「浅/深」+ 色系
    for src, dst in IRIS_LABEL_TO_COLOR.items():
        if src in label or label in src:
            return dst
    return None


def _side_metrics(side: Optional[dict]) -> dict[str, Any]:
    if not side:
        return {}
    lab = side.get("lab") or {}
    iris = side.get("iris_color") or {}
    return {
        "grade": grade_int_to_str(side.get("grade")),
        "lstar": lab.get("L"),
        "color": map_iris_color_label(iris.get("label")),
        "iris_color_label": iris.get("label"),
    }


def debug_run_image_urls(run_id: str, api_key: str) -> dict[str, str]:
    """Debug run 内原图 / 调色前后虹膜裁切 / 完整对比图 URL。"""
    base = f"/debug/files/{run_id}"
    key_q = f"?key={api_key}"
    return {
        "thumb_url": f"{base}/00_original.jpg{key_q}",
        "compare_url": f"{base}/07_sclera_before_after.jpg{key_q}",
        "thumb_before_url": f"{base}/08_iris_before.jpg{key_q}",
        "thumb_after_url": f"{base}/09_iris_after.jpg{key_q}",
        "viewer_url": f"/debug/viewer/{run_id}{key_q}",
    }


def experiment_snapshot_urls(record_id: int, api_key: str) -> dict[str, str]:
    """已持久化到实验快照目录的调色前后图 URL。"""
    base = f"/experiments/snapshots/{record_id}"
    key_q = f"?key={api_key}"
    return {
        "thumb_before_url": f"{base}/before.jpg{key_q}",
        "thumb_after_url": f"{base}/after.jpg{key_q}",
    }


def _split_compare_panel_array(
    img: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """从 07 对比图数组拆出左右面板。"""
    canvas_h = img.shape[0] - _COMPARE_FOOTER_H
    if canvas_h <= 0:
        return None, None
    canvas = img[:canvas_h]
    need_w = _COMPARE_PANEL_W + _COMPARE_GAP_W + _COMPARE_PANEL_W
    if canvas.shape[1] < need_w:
        return None, None
    left = canvas[:, :_COMPARE_PANEL_W]
    right = canvas[:, _COMPARE_PANEL_W + _COMPARE_GAP_W : need_w]
    return left, right


def _split_compare_panel(combined_path: Path) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """从 07 对比图拆出左右面板（兼容旧 debug run 无 08/09 的情况）。"""
    img = cv2.imread(str(combined_path))
    if img is None:
        return None, None
    return _split_compare_panel_array(img)


def _write_compare_pair(
    dest_dir: Path,
    before_img: np.ndarray,
    after_img: np.ndarray,
    record_id: int,
) -> dict[str, str]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest_dir / "before.jpg"), before_img)
    cv2.imwrite(str(dest_dir / "after.jpg"), after_img)
    return {
        "image_before_rel": f"{record_id}/before.jpg",
        "image_after_rel": f"{record_id}/after.jpg",
    }


def _imread_unicode(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def generate_compare_snapshots_from_image_rel(
    img_root: Path,
    snapshot_root: Path,
    record_id: int,
    image_rel: str,
    config: dict,
    config_path: Path,
) -> dict[str, Optional[str]]:
    """对 img/ 原图重新识别，生成并持久化调色前后虹膜裁切图。"""
    from app.services.debug_viz import build_debug_images
    from app.services.pipeline import AnalysisError, run_analysis

    rel = (image_rel or "").replace("\\", "/").lstrip("/")
    if not rel or ".." in rel:
        raise ValueError("invalid_image_rel")
    target = (img_root / rel).resolve()
    root_resolved = img_root.resolve()
    if not str(target).startswith(str(root_resolved)):
        raise ValueError("invalid_image_rel")
    if not target.is_file():
        raise ValueError("image_not_found")

    image_bgr = _imread_unicode(target)
    if image_bgr is None:
        raise ValueError("invalid_image_format")

    try:
        pipeline = run_analysis(image_bgr, config, config_path)
    except AnalysisError as exc:
        raise ValueError(str(exc)) from exc

    highlight_v = int(config.get("highlight_v_threshold", 240))
    eye_cfg = config.get("eye_closeup", {})
    images = build_debug_images(
        pipeline,
        eye_cfg,
        config=config,
        highlight_v=highlight_v,
    )
    dest_dir = snapshot_root / str(record_id)
    before = images.get("08_iris_before")
    after = images.get("09_iris_after")
    if before is not None and after is not None:
        return _write_compare_pair(dest_dir, before, after, record_id)

    panel = images.get("07_sclera_before_after")
    if panel is not None:
        left, right = _split_compare_panel_array(panel)
        if left is not None and right is not None:
            return _write_compare_pair(dest_dir, left, right, record_id)

    return {"image_before_rel": None, "image_after_rel": None}


def snapshot_compare_images(
    debug_output_root: Path,
    snapshot_root: Path,
    record_id: int,
    debug_run_id: str,
) -> dict[str, Optional[str]]:
    """
    将 debug run 的调色前后图复制到实验快照目录，便于长期保留。
    返回 image_before_rel / image_after_rel（相对 snapshot_root）。
    """
    if ".." in debug_run_id or "/" in debug_run_id or "\\" in debug_run_id:
        return {"image_before_rel": None, "image_after_rel": None}

    run_dir = debug_output_root / debug_run_id
    if not run_dir.is_dir():
        return {"image_before_rel": None, "image_after_rel": None}

    dest_dir = snapshot_root / str(record_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    before_dest = dest_dir / "before.jpg"
    after_dest = dest_dir / "after.jpg"

    before_src = run_dir / "08_iris_before.jpg"
    after_src = run_dir / "09_iris_after.jpg"
    if before_src.exists() and after_src.exists():
        shutil.copy2(before_src, before_dest)
        shutil.copy2(after_src, after_dest)
    else:
        panel_path = run_dir / "07_sclera_before_after.jpg"
        if not panel_path.exists():
            return {"image_before_rel": None, "image_after_rel": None}
        left, right = _split_compare_panel(panel_path)
        if left is None or right is None:
            return {"image_before_rel": None, "image_after_rel": None}
        cv2.imwrite(str(before_dest), left)
        cv2.imwrite(str(after_dest), right)

    return {
        "image_before_rel": f"{record_id}/before.jpg",
        "image_after_rel": f"{record_id}/after.jpg",
    }


def metrics_to_import_payload(metrics: dict, run_id: Optional[str] = None) -> dict[str, Any]:
    """将 debug metrics 转为实验记录表单预填数据（run_id 可选，无磁盘保存时也可导入）。"""
    compare = metrics.get("color_correction") or {}
    before = _side_metrics(compare.get("before"))
    after = _side_metrics(compare.get("after"))

    # 无校正对比时，主 metrics 作为「调色后/当前识别结果」
    if not before.get("grade") and not before.get("lstar"):
        main = _side_metrics(metrics)
        if not after.get("grade"):
            after = main
        if not before.get("grade"):
            before = {}

    source_rel = metrics.get("source_rel") or metrics.get("image_rel")
    source_name = metrics.get("source_filename") or metrics.get("original_filename")

    urls = debug_run_image_urls(run_id, "") if run_id else {}
    thumb_url = urls.get("thumb_url")
    viewer_url = urls.get("viewer_url")

    return {
        "debug_run_id": run_id or None,
        "image_rel": source_rel,
        "source_filename": source_name,
        "grade_before": before.get("grade"),
        "lstar_before": before.get("lstar"),
        "grade_after": after.get("grade"),
        "lstar_after": after.get("lstar"),
        "color": after.get("color") or before.get("color"),
        "color_before": before.get("color"),
        "color_after": after.get("color"),
        "iris_color_label": after.get("iris_color_label") or before.get("iris_color_label"),
        "thumb_url": thumb_url,
        "thumb_before_url": urls.get("thumb_before_url"),
        "thumb_after_url": urls.get("thumb_after_url"),
        "compare_url": urls.get("compare_url"),
        "viewer_url": viewer_url,
        "manual_adjusted": bool(metrics.get("manual_adjusted")),
        "detection_method": metrics.get("detection_method"),
        "summary": _build_summary(before, after, source_rel, run_id),
    }


def enrich_import_payload_urls(payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    """补全缩略图/图片预览 URL（含无 debug run 时仅用 image_rel 的情况）。"""
    out = dict(payload)
    run_id = out.get("debug_run_id")
    if run_id:
        urls = debug_run_image_urls(run_id, api_key)
        for key in (
            "thumb_url",
            "thumb_before_url",
            "thumb_after_url",
            "compare_url",
            "viewer_url",
        ):
            if urls.get(key):
                out[key] = urls[key]
    if out.get("image_rel"):
        out["image_url"] = f"/debug/img/file?rel={out['image_rel']}&key={api_key}"
        if not out.get("thumb_url"):
            out["thumb_url"] = out["image_url"]
    return out


def _build_summary(before: dict, after: dict, source_rel: Optional[str], run_id: Optional[str]) -> str:
    parts = []
    if run_id:
        parts.append(f"debug:{run_id}")
    elif source_rel:
        parts.append("debug:session")
    if source_rel:
        parts.append(f"img:{source_rel}")
    if before.get("grade"):
        parts.append(f"前 {before['grade']} L*={before.get('lstar')}")
    if after.get("grade"):
        parts.append(f"后 {after['grade']} L*={after.get('lstar')}")
    return " | ".join(parts)


def load_debug_run_metrics(debug_output_root: Path, run_id: str) -> Optional[dict]:
    if ".." in run_id or "/" in run_id or "\\" in run_id:
        return None
    metrics_path = debug_output_root / run_id / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def list_debug_runs_summary(debug_output_root: Path, api_key: str, limit: int = 30) -> list[dict]:
    if not debug_output_root.exists():
        return []
    dirs = sorted(
        [p for p in debug_output_root.iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )[:limit]
    out: list[dict] = []
    for run_dir in dirs:
        run_id = run_dir.name
        metrics = load_debug_run_metrics(debug_output_root, run_id)
        if not metrics:
            out.append({
                "run_id": run_id,
                "thumb_url": f"/debug/files/{run_id}/00_original.jpg?key={api_key}",
                "viewer_url": f"/debug/viewer/{run_id}?key={api_key}",
                "has_metrics": False,
            })
            continue
        payload = metrics_to_import_payload(metrics, run_id)
        out.append({
            "run_id": run_id,
            "has_metrics": True,
            "thumb_url": f"/debug/files/{run_id}/00_original.jpg?key={api_key}",
            "viewer_url": f"/debug/viewer/{run_id}?key={api_key}",
            "image_rel": payload.get("image_rel"),
            "source_filename": payload.get("source_filename"),
            "grade_before": payload.get("grade_before"),
            "lstar_before": payload.get("lstar_before"),
            "grade_after": payload.get("grade_after"),
            "lstar_after": payload.get("lstar_after"),
            "color": payload.get("color"),
            "summary": payload.get("summary"),
            "manual_adjusted": payload.get("manual_adjusted"),
        })
    return out
