"""实验记录管理 API（本地开发用，需 experiments.enabled=true）。"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

from app.services.experiment_debug_import import (
    COMPARE_AUDIT_FILENAME,
    enrich_audit_with_current_standards,
    enrich_import_payload_urls,
    experiment_snapshot_urls,
    generate_compare_snapshots_from_image_rel,
    list_debug_runs_summary,
    load_compare_audit,
    load_debug_run_metrics,
    metrics_to_import_payload,
    rebuild_compare_audit,
    snapshot_compare_images,
)
from app.services.experiment_store import ExperimentStore, create_experiment_store
from app.services.pipeline import load_config

router = APIRouter(prefix="/experiments", tags=["experiments"])

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "config" / "grade_thresholds.yaml"
IMG_ROOT = ROOT.parent / "img"

_store: Optional[ExperimentStore] = None
_store_key: Optional[str] = None


def _verify_key(key: Optional[str]) -> dict:
    config = load_config(CONFIG_PATH)
    exp_cfg = config.get("experiments", {})
    if not exp_cfg.get("enabled", False):
        raise HTTPException(status_code=404, detail="experiments_disabled")
    if key != exp_cfg.get("api_key", "iris-color-dev"):
        raise HTTPException(status_code=403, detail="invalid_experiments_key")
    return config


def _store_cache_key(exp_cfg: dict) -> str:
    backend = exp_cfg.get("backend", "sqlite")
    if backend == "mysql":
        mysql = exp_cfg.get("mysql") or {}
        return "mysql:{host}:{port}:{database}".format(
            host=mysql.get("host", "127.0.0.1"),
            port=mysql.get("port", 3306),
            database=mysql.get("database", "iris_experiment"),
        )
    return "sqlite:" + str(exp_cfg.get("db_path", "data/experiment_records.db"))


def _get_store(config: dict) -> ExperimentStore:
    global _store, _store_key
    exp_cfg = config.get("experiments", {})
    key = _store_cache_key(exp_cfg)
    if _store is None or _store_key != key:
        try:
            _store = create_experiment_store(exp_cfg, ROOT)
            _store_key = key
        except RuntimeError as exc:
            if str(exc) == "pymysql_not_installed":
                raise HTTPException(status_code=500, detail="pymysql_not_installed") from exc
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"database_unavailable: {exc}") from exc
    return _store


def _handle_value_error(exc: ValueError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def _debug_output_root(config: dict) -> Path:
    return ROOT / config.get("debug", {}).get("output_dir", "debug_output")


def _debug_api_key(config: dict) -> str:
    return config.get("debug", {}).get("api_key", "iris-color-dev")


def _experiment_snapshot_root(config: dict) -> Path:
    exp_cfg = config.get("experiments", {})
    rel = exp_cfg.get("snapshot_dir", "data/experiment_snapshots")
    return ROOT / rel


def _apply_compare_snapshot_paths(
    store: ExperimentStore,
    record: dict,
    paths: dict[str, Optional[str]],
) -> dict:
    if not paths.get("image_before_rel"):
        return record
    updated = store.update_record_images(
        int(record["id"]),
        image_before_rel=paths["image_before_rel"],
        image_after_rel=paths["image_after_rel"],
    )
    return updated or record


def _persist_compare_snapshots(config: dict, store: ExperimentStore, record: dict) -> dict:
    """保存实验记录后，持久化调色前后对比图（debug run 或原图重识别）。"""
    if record.get("image_before_rel") and record.get("image_after_rel"):
        return record

    record_id = int(record["id"])
    snapshot_root = _experiment_snapshot_root(config)

    run_id = record.get("debug_run_id")
    if run_id:
        paths = snapshot_compare_images(
            _debug_output_root(config),
            snapshot_root,
            record_id,
            run_id,
            config=config,
            record=record,
        )
        if paths.get("image_before_rel"):
            return _apply_compare_snapshot_paths(store, record, paths)

    image_rel = record.get("image_rel")
    if image_rel:
        try:
            paths = generate_compare_snapshots_from_image_rel(
                IMG_ROOT,
                snapshot_root,
                record_id,
                image_rel,
                config,
                CONFIG_PATH,
                record=record,
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        if paths.get("image_before_rel"):
            return _apply_compare_snapshot_paths(store, record, paths)
        raise ValueError("compare_images_unavailable")

    return record


def ensure_record_compare_images(
    config: dict,
    store: ExperimentStore,
    record_id: int,
) -> dict:
    """确保记录已有调色前后快照；若无则按 debug run 或 image_rel 生成。"""
    record = store.get_by_id(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="record_not_found")

    snapshot_root = _experiment_snapshot_root(config)
    try:
        if record.get("image_before_rel") and record.get("image_after_rel"):
            out = dict(record)
        else:
            out = _persist_compare_snapshots(config, store, record)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not out.get("image_before_rel"):
        if not record.get("image_rel") and not record.get("debug_run_id"):
            raise HTTPException(status_code=422, detail="no_image_source")
        raise HTTPException(status_code=422, detail="compare_images_unavailable")

    audit = load_compare_audit(snapshot_root, record_id)
    if not audit:
        audit = rebuild_compare_audit(
            snapshot_root=snapshot_root,
            record_id=record_id,
            record=out,
            config=config,
            config_path=CONFIG_PATH,
            img_root=IMG_ROOT,
            debug_output_root=_debug_output_root(config),
        )
    if audit:
        audit = enrich_audit_with_current_standards(audit, config)

    urls = experiment_snapshot_urls(
        record_id,
        config.get("experiments", {}).get("api_key", "iris-color-dev"),
    )
    api_key = config.get("experiments", {}).get("api_key", "iris-color-dev")
    result = {
        **out,
        "thumb_before_url": urls["thumb_before_url"],
        "thumb_after_url": urls["thumb_after_url"],
        "compare_audit": audit,
    }
    if audit:
        result["audit_url"] = f"/experiments/snapshots/{record_id}/{COMPARE_AUDIT_FILENAME}?key={api_key}"
    return result


@router.get("/ui", response_class=HTMLResponse)
def experiment_ui(key: str = Query(...)):
    _verify_key(key)
    html_path = ROOT / "app" / "static" / "experiment_ui.html"
    if not html_path.exists():
        raise HTTPException(status_code=500, detail="experiment_ui_missing")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@router.get("/records")
def list_records(
    key: str = Query(...),
    group_name: Optional[str] = Query(None),
    subgroup_name: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    operator: Optional[str] = Query(None),
    camera_device: Optional[str] = Query(None),
    light_device: Optional[str] = Query(None),
    color: Optional[str] = Query(None),
    grade_before: Optional[str] = Query(None),
    grade_after: Optional[str] = Query(None),
):
    config = _verify_key(key)
    store = _get_store(config)
    records = store.list_records(
        group_name=group_name,
        subgroup_name=subgroup_name,
        date_from=date_from,
        date_to=date_to,
        operator=operator,
        camera_device=camera_device,
        light_device=light_device,
        color=color,
        grade_before=grade_before,
        grade_after=grade_after,
    )
    return {"records": records, "count": len(records)}


@router.get("/records/{record_id}")
def get_record(record_id: int, key: str = Query(...)):
    config = _verify_key(key)
    store = _get_store(config)
    record = store.get_by_id(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="record_not_found")
    return record


@router.post("/records/{record_id}/ensure-compare-images")
def ensure_compare_images(record_id: int, key: str = Query(...)):
    """按需生成或返回实验记录的调色前后对比图。"""
    config = _verify_key(key)
    store = _get_store(config)
    try:
        return ensure_record_compare_images(config, store, record_id)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.post("/records", status_code=201)
def create_record(key: str = Query(...), payload: dict = Body(...)):
    config = _verify_key(key)
    store = _get_store(config)
    try:
        record = store.create_record(payload)
        record = _persist_compare_snapshots(config, store, record)
        return record
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.put("/records/{record_id}")
def update_record(record_id: int, key: str = Query(...), payload: dict = Body(...)):
    config = _verify_key(key)
    store = _get_store(config)
    try:
        record = store.update_record(record_id, payload)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    if not record:
        raise HTTPException(status_code=404, detail="record_not_found")
    record = _persist_compare_snapshots(config, store, record)
    return record


@router.delete("/records/{record_id}")
def delete_record(record_id: int, key: str = Query(...)):
    config = _verify_key(key)
    store = _get_store(config)
    if not store.delete_record(record_id):
        raise HTTPException(status_code=404, detail="record_not_found")
    return {"ok": True, "id": record_id}


@router.post("/records/bulk-delete")
def bulk_delete(key: str = Query(...), payload: dict = Body(...)):
    config = _verify_key(key)
    store = _get_store(config)
    ids = payload.get("ids") or []
    if not isinstance(ids, list):
        raise HTTPException(status_code=400, detail="invalid_ids")
    try:
        id_list = [int(i) for i in ids]
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="invalid_ids") from exc
    deleted = store.bulk_delete(id_list)
    return {"ok": True, "deleted": deleted}


@router.get("/meta/options")
def meta_options(key: str = Query(...)):
    config = _verify_key(key)
    store = _get_store(config)
    return store.get_distinct_options()


@router.get("/meta/suggest-names")
def suggest_names(
    key: str = Query(...),
    group_name: Optional[str] = Query(None),
    subgroup_name: Optional[str] = Query(None),
    date: Optional[str] = Query(None, description="保留参数，中文组名不再绑定日期"),
):
    config = _verify_key(key)
    store = _get_store(config)
    date_yyyymmdd = date.replace("-", "") if date else None
    suggested_group = store.suggest_group_name(date_yyyymmdd)
    g = (group_name or "").strip() or suggested_group
    suggested_subgroup = (subgroup_name or "").strip() or (
        store.suggest_subgroup_name(g) if g else "第一小组"
    )
    defaults = store.get_subgroup_defaults(g, suggested_subgroup) if g and suggested_subgroup else {}
    return {
        "group_name": g,
        "subgroup_name": suggested_subgroup,
        "group_format": "第一大组、第二大组…",
        "subgroup_format": "第一小组、第二小组…（可留空）",
        "subgroup_defaults": defaults,
    }


@router.get("/meta/subgroup-defaults")
def subgroup_defaults(
    key: str = Query(...),
    group_name: str = Query(...),
    subgroup_name: str = Query(...),
):
    config = _verify_key(key)
    store = _get_store(config)
    return store.get_subgroup_defaults(group_name.strip(), subgroup_name.strip())


@router.post("/import/from-metrics")
def import_from_metrics(key: str = Query(...), body: dict = Body(...)):
    """从 Debug 当前识别 metrics 直接导入（无需 save_to_disk）。"""
    config = _verify_key(key)
    if not config.get("debug", {}).get("enabled", True):
        raise HTTPException(status_code=403, detail="debug_disabled")
    metrics = body.get("metrics")
    if not isinstance(metrics, dict):
        raise HTTPException(status_code=400, detail="missing_metrics")
    run_id = body.get("debug_run_id")
    if run_id is not None and (not isinstance(run_id, str) or ".." in run_id or "/" in run_id or "\\" in run_id):
        raise HTTPException(status_code=400, detail="invalid_run_id")
    source_rel = body.get("source_rel") or metrics.get("source_rel") or metrics.get("image_rel")
    merged = dict(metrics)
    if source_rel:
        merged["source_rel"] = source_rel
        merged["image_rel"] = source_rel
    source_filename = body.get("source_filename")
    if source_filename:
        merged["source_filename"] = source_filename
        merged["original_filename"] = source_filename
    if "skip_quality" in body:
        merged["skip_quality"] = bool(body.get("skip_quality"))
    if "manual_adjusted" in body:
        merged["manual_adjusted"] = bool(body.get("manual_adjusted"))
    payload = metrics_to_import_payload(merged, run_id or None)
    dbg_key = _debug_api_key(config)
    return enrich_import_payload_urls(payload, dbg_key)


@router.get("/debug/runs")
def list_debug_runs_for_import(
    key: str = Query(...),
    limit: int = Query(30, ge=1, le=100),
):
    """列出最近 debug 运行，供实验记录导入。"""
    config = _verify_key(key)
    if not config.get("debug", {}).get("enabled", True):
        raise HTTPException(status_code=403, detail="debug_disabled")
    dbg_key = _debug_api_key(config)
    runs = list_debug_runs_summary(_debug_output_root(config), dbg_key, limit)
    return {"runs": runs, "count": len(runs), "debug_ui_url": f"/debug/ui?key={dbg_key}"}


@router.get("/debug/runs/{run_id}/import")
def get_debug_import_payload(run_id: str, key: str = Query(...)):
    """读取 debug run 的 metrics，返回实验记录表单预填数据。"""
    config = _verify_key(key)
    if not config.get("debug", {}).get("enabled", True):
        raise HTTPException(status_code=403, detail="debug_disabled")
    if ".." in run_id or "/" in run_id or "\\" in run_id:
        raise HTTPException(status_code=400, detail="invalid_run_id")
    metrics = load_debug_run_metrics(_debug_output_root(config), run_id)
    if not metrics:
        raise HTTPException(status_code=404, detail="debug_run_not_found")
    dbg_key = _debug_api_key(config)
    payload = metrics_to_import_payload(metrics, run_id)
    return enrich_import_payload_urls(payload, dbg_key)


@router.get("/snapshots/{record_id}/{filename}")
def get_experiment_snapshot(
    record_id: int,
    filename: str,
    key: str = Query(...),
):
    """读取实验记录持久化的调色前/后快照或审计 JSON。"""
    config = _verify_key(key)
    allowed = ("before.jpg", "after.jpg", COMPARE_AUDIT_FILENAME)
    if filename not in allowed:
        raise HTTPException(status_code=400, detail="invalid_snapshot_file")
    path = _experiment_snapshot_root(config) / str(record_id) / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="snapshot_not_found")
    if filename == COMPARE_AUDIT_FILENAME:
        audit = load_compare_audit(_experiment_snapshot_root(config), record_id)
        if not audit:
            raise HTTPException(status_code=404, detail="audit_not_found")
        return enrich_audit_with_current_standards(audit, config)
    return FileResponse(path, media_type="image/jpeg")
