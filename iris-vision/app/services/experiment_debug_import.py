"""从 debug 运行结果映射到实验记录字段。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from app.services.experiment_store import COLOR_VALUES

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

    thumb_url = f"/debug/files/{run_id}/00_original.jpg" if run_id else None
    viewer_url = f"/debug/viewer/{run_id}" if run_id else None

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
        if out.get("thumb_url") and "?" not in out["thumb_url"]:
            out["thumb_url"] = f"{out['thumb_url']}?key={api_key}"
        if out.get("viewer_url") and "?" not in out["viewer_url"]:
            out["viewer_url"] = f"{out['viewer_url']}?key={api_key}"
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
