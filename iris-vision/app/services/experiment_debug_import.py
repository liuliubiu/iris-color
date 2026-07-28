"""从 debug 运行结果映射到实验记录字段。"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from app.services.experiment_store import COLOR_VALUES

_COMPARE_PANEL_W = 420
_COMPARE_GAP_W = 12
_COMPARE_FOOTER_H = 56
COMPARE_AUDIT_FILENAME = "compare_audit.json"

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


def experiment_snapshot_urls(record_id: int, api_key: str, *, table_set: str = "prod") -> dict[str, str]:
    """已持久化到实验快照目录的调色前后图 URL。"""
    prefix = "/experiments/test/snapshots" if table_set == "test" else "/experiments/snapshots"
    base = f"{prefix}/{record_id}"
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
    audit: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest_dir / "before.jpg"), before_img)
    cv2.imwrite(str(dest_dir / "after.jpg"), after_img)
    if audit is not None:
        save_compare_audit(dest_dir, audit)
    return {
        "image_before_rel": f"{record_id}/before.jpg",
        "image_after_rel": f"{record_id}/after.jpg",
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _standards_snapshot(config: dict) -> dict[str, Any]:
    sn = config.get("sclera_normalization", {}) or {}
    grade = config.get("grade", {}) or {}
    return {
        "grade_boundaries": grade.get("boundaries"),
        "sclera_algorithm_version": sn.get("algorithm_version", "legacy"),
        "sclera_enabled": bool(sn.get("enabled", False)),
        "sclera_target_l": sn.get("target_l"),
        "sclera_normalize_luminance": sn.get("normalize_luminance"),
        "sclera_normalize_chroma": sn.get("normalize_chroma"),
        "sclera_luminance_strength": sn.get("luminance_strength"),
        "sclera_chroma_strength": sn.get("chroma_strength"),
        "sclera_adaptive_strength": sn.get("adaptive_strength"),
        "sclera_quality_min_apply": sn.get("quality_min_apply"),
        "sclera_quality_strength_power": sn.get("quality_strength_power"),
        "sclera_luminance_log_deadband": sn.get("luminance_log_deadband"),
        "sclera_chroma_log_deadband": sn.get("chroma_log_deadband"),
        "sclera_max_gain_ratio": sn.get("max_gain_ratio"),
        "sclera_preserve_luminance": sn.get("preserve_iris_luminance_during_chroma"),
        "sclera_device_luminance_profiles": sn.get("device_luminance_profiles"),
    }


def _grade_match(record_grade: Optional[str], side: Optional[dict]) -> Optional[bool]:
    if not record_grade or not side:
        return None
    return record_grade == grade_int_to_str(side.get("grade"))


def _lstar_match(record_lstar: Any, side: Optional[dict], tol: float = 0.05) -> Optional[bool]:
    if record_lstar is None or record_lstar == "" or not side:
        return None
    lab = side.get("lab") or {}
    lab_l = lab.get("L")
    if lab_l is None:
        return None
    try:
        return abs(float(record_lstar) - float(lab_l)) <= tol
    except (TypeError, ValueError):
        return None


def build_compare_audit(
    *,
    source: str,
    metrics: dict,
    config: dict,
    record: Optional[dict] = None,
    debug_run_id: Optional[str] = None,
    skip_quality_used: bool = False,
) -> dict[str, Any]:
    """生成对比快照审计：巩膜校正参数、调色前后数值、与实验记录一致性。"""
    cc = metrics.get("color_correction") or {}
    sn = metrics.get("sclera_normalization") or {}
    record_values = None
    consistency = None
    if record:
        record_values = {
            "grade_before": record.get("grade_before"),
            "lstar_before": record.get("lstar_before"),
            "grade_after": record.get("grade_after"),
            "lstar_after": record.get("lstar_after"),
        }
        before = cc.get("before") or {}
        after = cc.get("after") or {}
        consistency = {
            "before_grade_match": _grade_match(record.get("grade_before"), before),
            "before_lstar_match": _lstar_match(record.get("lstar_before"), before),
            "after_grade_match": _grade_match(record.get("grade_after"), after),
            "after_lstar_match": _lstar_match(record.get("lstar_after"), after),
        }
    return {
        "generated_at": _utc_now_iso(),
        "source": source,
        "debug_run_id": debug_run_id,
        "algorithm": "run_analysis + compute_color_correction_compare (与 Debug 相同)",
        "sclera_correction_applied": bool(cc.get("applied")),
        "sclera_status": cc.get("status") or sn.get("status"),
        "color_correction": cc,
        "sclera_normalization": sn,
        "standards_at_generation": _standards_snapshot(config),
        "record_values_at_generation": record_values,
        "consistency_with_record": consistency,
        "manual_adjusted": bool(metrics.get("manual_adjusted")),
        "detection_method": metrics.get("detection_method"),
        "skip_quality_used": skip_quality_used,
    }


def save_compare_audit(dest_dir: Path, audit: dict[str, Any]) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / COMPARE_AUDIT_FILENAME).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_compare_audit(snapshot_root: Path, record_id: int) -> Optional[dict[str, Any]]:
    path = snapshot_root / str(record_id) / COMPARE_AUDIT_FILENAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def enrich_audit_with_current_standards(audit: dict[str, Any], config: dict) -> dict[str, Any]:
    current = _standards_snapshot(config)
    stored = audit.get("standards_at_generation") or {}
    out = dict(audit)
    out["standards_current"] = current
    out["standards_changed_since_generation"] = stored != current
    return out


def rebuild_compare_audit(
    *,
    snapshot_root: Path,
    record_id: int,
    record: dict,
    config: dict,
    config_path: Path,
    img_root: Path,
    debug_output_root: Path,
) -> Optional[dict[str, Any]]:
    """快照图已存在但缺少 audit 时，从 debug metrics 或重算 metrics 补写审计。"""
    dest_dir = snapshot_root / str(record_id)
    if not (dest_dir / "before.jpg").is_file():
        return None

    debug_run_id = record.get("debug_run_id")
    if debug_run_id:
        metrics = load_debug_run_metrics(debug_output_root, debug_run_id)
        if metrics:
            audit = build_compare_audit(
                source="debug_run",
                metrics=metrics,
                config=config,
                record=record,
                debug_run_id=debug_run_id,
            )
            save_compare_audit(dest_dir, audit)
            return audit

    image_rel = record.get("image_rel")
    if not image_rel:
        return None

    from app.services.debug_viz import build_debug_metrics
    from app.services.pipeline import AnalysisError, run_analysis

    rel = (image_rel or "").replace("\\", "/").lstrip("/")
    if not rel or ".." in rel:
        return None
    target = (img_root / rel).resolve()
    if not str(target).startswith(str(img_root.resolve())) or not target.is_file():
        return None
    image_bgr = _imread_unicode(target)
    if image_bgr is None:
        return None
    try:
        pipeline, skip_quality_used = _run_analysis_for_compare(image_bgr, config, config_path)
    except ValueError:
        return None

    highlight_v = int(config.get("highlight_v_threshold", 240))
    metrics = build_debug_metrics(pipeline, highlight_v, config=config)
    audit = build_compare_audit(
        source="reanalysis_metrics_only",
        metrics=metrics,
        config=config,
        record=record,
        skip_quality_used=skip_quality_used,
    )
    save_compare_audit(dest_dir, audit)
    return audit


def _run_analysis_for_compare(
    image_bgr: np.ndarray,
    config: dict,
    config_path: Path,
    *,
    skip_quality: Optional[bool] = None,
    manual_detection: Optional[dict] = None,
    closeup_mode: str = "auto",
):
    """实验对比重算：质量门槛未通过时自动 skip_quality（与 Debug 勾选跳过一致）。"""
    from app.services.pipeline import AnalysisError, run_analysis

    run_kwargs: dict[str, Any] = {
        "closeup_mode": closeup_mode if closeup_mode in ("auto", "precise", "rough") else "auto",
    }
    if manual_detection:
        run_kwargs["manual_detection"] = manual_detection
    if skip_quality is True:
        run_kwargs["skip_quality"] = True
        return run_analysis(image_bgr, config, config_path, **run_kwargs), True

    try:
        if skip_quality is False:
            run_kwargs["skip_quality"] = False
        return run_analysis(image_bgr, config, config_path, **run_kwargs), False
    except AnalysisError as exc:
        if exc.code == "quality_check_failed":
            run_kwargs["skip_quality"] = True
            return run_analysis(image_bgr, config, config_path, **run_kwargs), True
        raise ValueError(exc.code) from exc


def _parse_manual_params_dict(raw: Any) -> Optional[dict[str, float]]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("invalid_manual_params")
    required = ["center_x", "center_y", "pupil_radius", "inner_radius", "outer_radius"]
    if any(key not in raw for key in required):
        raise ValueError("manual_params_missing_fields")
    try:
        return {key: float(raw[key]) for key in required}
    except (TypeError, ValueError) as exc:
        raise ValueError("manual_params_must_be_numbers") from exc


def load_record_image_bgr(
    record: dict,
    img_root: Path,
    debug_output_root: Path,
) -> tuple[np.ndarray, Optional[str]]:
    """从 image_rel 或 debug run 原图加载 BGR 图像。"""
    image_rel = record.get("image_rel")
    if image_rel:
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
        return image_bgr, rel

    debug_run_id = record.get("debug_run_id")
    if debug_run_id:
        if ".." in debug_run_id or "/" in debug_run_id or "\\" in debug_run_id:
            raise ValueError("invalid_debug_run_id")
        original_path = debug_output_root / debug_run_id / "00_original.jpg"
        if not original_path.is_file():
            raise ValueError("debug_original_not_found")
        image_bgr = _imread_unicode(original_path)
        if image_bgr is None:
            raise ValueError("invalid_image_format")
        return image_bgr, None

    raise ValueError("no_image_source")


def run_record_recognition(
    image_bgr: np.ndarray,
    config: dict,
    config_path: Path,
    *,
    image_rel: Optional[str] = None,
    skip_quality: Optional[bool] = None,
    manual_detection: Optional[dict] = None,
    closeup_mode: str = "auto",
) -> tuple[Any, dict, bool]:
    """对实验记录关联原图运行识别 pipeline，返回 (pipeline, metrics, skip_quality_used)。"""
    from app.services.debug_viz import build_debug_metrics

    pipeline, skip_quality_used = _run_analysis_for_compare(
        image_bgr,
        config,
        config_path,
        skip_quality=skip_quality,
        manual_detection=manual_detection,
        closeup_mode=closeup_mode,
    )
    highlight_v = int(config.get("highlight_v_threshold", 240))
    metrics = build_debug_metrics(pipeline, highlight_v, config=config)
    if image_rel:
        metrics["source_rel"] = image_rel
        metrics["image_rel"] = image_rel
    metrics["skip_quality"] = bool(skip_quality_used if skip_quality is None else skip_quality)
    return pipeline, metrics, skip_quality_used


def build_record_recognition_preview(
    record: dict,
    config: dict,
    config_path: Path,
    img_root: Path,
    debug_output_root: Path,
    *,
    skip_quality: Optional[bool] = None,
    manual_params: Any = None,
    closeup_mode: str = "auto",
) -> dict[str, Any]:
    """识别预览：返回 metrics、新旧对比、可视化 base64（不写入 DB）。"""
    from app.services.debug_viz import build_debug_images, images_to_base64

    image_bgr, image_rel = load_record_image_bgr(record, img_root, debug_output_root)
    manual_detection = _parse_manual_params_dict(manual_params)
    if skip_quality is None:
        skip_quality = bool(record.get("skip_quality"))

    pipeline, metrics, skip_quality_used = run_record_recognition(
        image_bgr,
        config,
        config_path,
        image_rel=image_rel,
        skip_quality=skip_quality,
        manual_detection=manual_detection,
        closeup_mode=closeup_mode,
    )

    highlight_v = int(config.get("highlight_v_threshold", 240))
    eye_cfg = config.get("eye_closeup", {})
    images = build_debug_images(pipeline, eye_cfg, config=config, highlight_v=highlight_v)
    preview_images = {
        "00_original": image_bgr,
        "02_iris_ring": images.get("02_iris_ring"),
        "04_valid_samples": images.get("04_valid_samples"),
    }
    preview_images = {k: v for k, v in preview_images.items() if v is not None}

    import_payload = metrics_to_import_payload(metrics, record.get("debug_run_id"))
    current = {
        "grade_before": record.get("grade_before"),
        "lstar_before": record.get("lstar_before"),
        "grade_after": record.get("grade_after"),
        "lstar_after": record.get("lstar_after"),
        "color": record.get("color"),
        "skip_quality": bool(record.get("skip_quality")),
        "manual_adjusted": bool(record.get("manual_adjusted")),
    }
    proposed = {
        "grade_before": import_payload.get("grade_before"),
        "lstar_before": import_payload.get("lstar_before"),
        "grade_after": import_payload.get("grade_after"),
        "lstar_after": import_payload.get("lstar_after"),
        "color": import_payload.get("color"),
        "skip_quality": bool(metrics.get("skip_quality")),
        "manual_adjusted": bool(metrics.get("manual_adjusted")),
    }
    mp = metrics.get("manual_params")
    if not mp and metrics.get("pupil_center"):
        mp = {
            "center_x": metrics["pupil_center"][0],
            "center_y": metrics["pupil_center"][1],
            "pupil_radius": metrics.get("pupil_radius"),
            "inner_radius": metrics.get("inner_radius"),
            "outer_radius": metrics.get("outer_radius"),
        }

    return {
        "metrics": metrics,
        "import_payload": import_payload,
        "current_record": current,
        "proposed_record": proposed,
        "manual_params": mp,
        "skip_quality_used": skip_quality_used,
        "image_rel": image_rel,
        "images_base64": images_to_base64(preview_images),
    }


def _imread_unicode(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _generate_compare_snapshots_from_pipeline(
    snapshot_root: Path,
    record_id: int,
    pipeline,
    image_bgr: np.ndarray,
    config: dict,
    *,
    record: Optional[dict] = None,
    skip_quality_used: bool = False,
    source: str = "reanalysis",
    debug_run_id: Optional[str] = None,
) -> dict[str, Optional[str]]:
    from app.services.debug_viz import build_debug_images, build_debug_metrics

    highlight_v = int(config.get("highlight_v_threshold", 240))
    eye_cfg = config.get("eye_closeup", {})
    metrics = build_debug_metrics(pipeline, highlight_v, config=config)
    audit = build_compare_audit(
        source=source,
        metrics=metrics,
        config=config,
        record=record,
        debug_run_id=debug_run_id,
        skip_quality_used=skip_quality_used,
    )
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
        return _write_compare_pair(dest_dir, before, after, record_id, audit=audit)

    panel = images.get("07_sclera_before_after")
    if panel is not None:
        left, right = _split_compare_panel_array(panel)
        if left is not None and right is not None:
            return _write_compare_pair(dest_dir, left, right, record_id, audit=audit)

    return {"image_before_rel": None, "image_after_rel": None}


def generate_compare_snapshots_from_image_rel(
    img_root: Path,
    snapshot_root: Path,
    record_id: int,
    image_rel: str,
    config: dict,
    config_path: Path,
    record: Optional[dict] = None,
    *,
    skip_quality: Optional[bool] = None,
    manual_detection: Optional[dict] = None,
    closeup_mode: str = "auto",
    force: bool = False,
) -> dict[str, Optional[str]]:
    """对 img/ 原图重新识别，生成并持久化调色前后虹膜裁切图及审计。"""
    if not force and record and record.get("image_before_rel") and record.get("image_after_rel"):
        return {
            "image_before_rel": record.get("image_before_rel"),
            "image_after_rel": record.get("image_after_rel"),
        }

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

    if skip_quality is None and record is not None:
        skip_quality = bool(record.get("skip_quality"))

    try:
        pipeline, skip_quality_used = _run_analysis_for_compare(
            image_bgr,
            config,
            config_path,
            skip_quality=skip_quality,
            manual_detection=manual_detection,
            closeup_mode=closeup_mode,
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    return _generate_compare_snapshots_from_pipeline(
        snapshot_root,
        record_id,
        pipeline,
        image_bgr,
        config,
        record=record,
        skip_quality_used=skip_quality_used,
        source="reanalysis",
    )


def generate_compare_snapshots_for_record(
    record: dict,
    snapshot_root: Path,
    record_id: int,
    config: dict,
    config_path: Path,
    img_root: Path,
    debug_output_root: Path,
    *,
    skip_quality: Optional[bool] = None,
    manual_detection: Optional[dict] = None,
    closeup_mode: str = "auto",
    force: bool = False,
) -> dict[str, Optional[str]]:
    """按实验记录关联原图（image_rel 或 debug run）重算并持久化对比快照。"""
    if not force and record.get("image_before_rel") and record.get("image_after_rel"):
        return {
            "image_before_rel": record.get("image_before_rel"),
            "image_after_rel": record.get("image_after_rel"),
        }

    image_bgr, image_rel = load_record_image_bgr(record, img_root, debug_output_root)
    if skip_quality is None:
        skip_quality = bool(record.get("skip_quality"))

    try:
        pipeline, skip_quality_used = _run_analysis_for_compare(
            image_bgr,
            config,
            config_path,
            skip_quality=skip_quality,
            manual_detection=manual_detection,
            closeup_mode=closeup_mode,
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    source = "reanalysis"
    debug_run_id = record.get("debug_run_id")
    if not image_rel:
        source = "reanalysis_from_debug_original"

    return _generate_compare_snapshots_from_pipeline(
        snapshot_root,
        record_id,
        pipeline,
        image_bgr,
        config,
        record=record,
        skip_quality_used=skip_quality_used,
        source=source,
        debug_run_id=debug_run_id if not image_rel else None,
    )


def apply_record_recognition(
    record: dict,
    store,
    config: dict,
    config_path: Path,
    img_root: Path,
    debug_output_root: Path,
    snapshot_root: Path,
    *,
    overwrite_record: bool,
    overwrite_snapshots: bool,
    skip_quality: Optional[bool] = None,
    manual_params: Any = None,
    closeup_mode: str = "auto",
) -> dict[str, Any]:
    """应用重新识别结果：可选覆盖 DB 字段与对比快照。"""
    manual_detection = _parse_manual_params_dict(manual_params)
    if skip_quality is None:
        skip_quality = bool(record.get("skip_quality"))

    image_bgr, image_rel = load_record_image_bgr(record, img_root, debug_output_root)
    pipeline, metrics, skip_quality_used = run_record_recognition(
        image_bgr,
        config,
        config_path,
        image_rel=image_rel or record.get("image_rel"),
        skip_quality=skip_quality,
        manual_detection=manual_detection,
        closeup_mode=closeup_mode,
    )
    import_payload = metrics_to_import_payload(metrics, record.get("debug_run_id"))
    record_id = int(record["id"])
    updated = dict(record)

    if overwrite_record:
        patch = {
            "group_name": record["group_name"],
            "subgroup_name": record.get("subgroup_name"),
            "experiment_date": record["experiment_date"],
            "operator": record["operator"],
            "camera_device": record.get("camera_device"),
            "light_device": record.get("light_device"),
            "illuminance": record.get("illuminance"),
            "color": import_payload.get("color") or record.get("color"),
            "grade_before": import_payload.get("grade_before"),
            "lstar_before": import_payload.get("lstar_before"),
            "grade_after": import_payload.get("grade_after"),
            "lstar_after": import_payload.get("lstar_after"),
            "notes": record.get("notes"),
            "image_rel": record.get("image_rel"),
            "debug_run_id": record.get("debug_run_id"),
            "skip_quality": bool(metrics.get("skip_quality")),
            "manual_adjusted": bool(metrics.get("manual_adjusted")),
            "include_in_stats": record.get("include_in_stats", True),
        }
        updated = store.update_record(record_id, patch) or updated

    if overwrite_snapshots:
        paths = _generate_compare_snapshots_from_pipeline(
            snapshot_root,
            record_id,
            pipeline,
            image_bgr,
            config,
            record=updated if overwrite_record else record,
            skip_quality_used=skip_quality_used,
            source="reanalysis",
            debug_run_id=record.get("debug_run_id") if not record.get("image_rel") else None,
        )
        if paths.get("image_before_rel"):
            updated = store.update_record_images(
                record_id,
                image_before_rel=paths["image_before_rel"],
                image_after_rel=paths["image_after_rel"],
            ) or updated

    return {
        "record": updated,
        "metrics": metrics,
        "import_payload": import_payload,
        "skip_quality_used": skip_quality_used,
        "overwritten_record": overwrite_record,
        "overwritten_snapshots": overwrite_snapshots,
    }


def bulk_apply_record_recognition(
    record_ids: list[int],
    store,
    config: dict,
    config_path: Path,
    img_root: Path,
    debug_output_root: Path,
    snapshot_root: Path,
    *,
    overwrite_record: bool = True,
    overwrite_snapshots: bool = True,
    skip_quality: Optional[bool] = None,
    closeup_mode: str = "auto",
) -> dict[str, Any]:
    """批量应用重新识别结果。"""
    results: list[dict[str, Any]] = []
    succeeded = 0
    failed = 0
    for rid in record_ids:
        record = store.get_by_id(int(rid))
        if not record:
            results.append({"id": int(rid), "ok": False, "error": "record_not_found"})
            failed += 1
            continue
        if not record.get("image_rel") and not record.get("debug_run_id"):
            results.append({"id": int(rid), "ok": False, "error": "no_image_source"})
            failed += 1
            continue
        try:
            out = apply_record_recognition(
                record,
                store,
                config,
                config_path,
                img_root,
                debug_output_root,
                snapshot_root,
                overwrite_record=overwrite_record,
                overwrite_snapshots=overwrite_snapshots,
                skip_quality=skip_quality,
                closeup_mode=closeup_mode,
            )
            results.append({"id": int(rid), "ok": True, "record": out["record"]})
            succeeded += 1
        except Exception as exc:
            results.append({"id": int(rid), "ok": False, "error": str(exc)})
            failed += 1
    return {
        "results": results,
        "succeeded": succeeded,
        "failed": failed,
        "total": len(record_ids),
    }


def snapshot_compare_images(
    debug_output_root: Path,
    snapshot_root: Path,
    record_id: int,
    debug_run_id: str,
    *,
    config: Optional[dict] = None,
    record: Optional[dict] = None,
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

    metrics = load_debug_run_metrics(debug_output_root, debug_run_id) if config else None
    audit = None
    if metrics and config:
        audit = build_compare_audit(
            source="debug_run",
            metrics=metrics,
            config=config,
            record=record,
            debug_run_id=debug_run_id,
        )

    before_src = run_dir / "08_iris_before.jpg"
    after_src = run_dir / "09_iris_after.jpg"
    if before_src.exists() and after_src.exists():
        shutil.copy2(before_src, before_dest)
        shutil.copy2(after_src, after_dest)
        if audit:
            save_compare_audit(dest_dir, audit)
        return {
            "image_before_rel": f"{record_id}/before.jpg",
            "image_after_rel": f"{record_id}/after.jpg",
        }

    panel_path = run_dir / "07_sclera_before_after.jpg"
    if panel_path.exists():
        left, right = _split_compare_panel(panel_path)
        if left is not None and right is not None:
            cv2.imwrite(str(before_dest), left)
            cv2.imwrite(str(after_dest), right)
            if audit:
                save_compare_audit(dest_dir, audit)
            return {
                "image_before_rel": f"{record_id}/before.jpg",
                "image_after_rel": f"{record_id}/after.jpg",
            }

    return {"image_before_rel": None, "image_after_rel": None}


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
        "skip_quality": bool(metrics.get("skip_quality")),
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
