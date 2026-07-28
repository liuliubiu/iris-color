"""实验记录管理 API（本地开发用，需 experiments.enabled=true）。"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

from app.services.experiment_debug_import import (
    COMPARE_AUDIT_FILENAME,
    apply_record_recognition,
    bulk_apply_record_recognition,
    build_record_recognition_preview,
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
from app.services.experiment_stability import analyze_stability
from app.services.experiment_store import ExperimentStore, TABLE_PROD, TABLE_TEST, create_experiment_store
from app.services.pipeline import load_config

router = APIRouter(prefix="/experiments", tags=["experiments"])

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "config" / "grade_thresholds.yaml"
IMG_ROOT = ROOT.parent / "img"

_store: Optional[ExperimentStore] = None
_store_key: Optional[str] = None
_stores: dict[str, ExperimentStore] = {}


def _parse_table_set(table_set: Optional[str]) -> str:
    value = (table_set or "prod").strip().lower()
    if value not in ("prod", "test"):
        raise HTTPException(status_code=400, detail="invalid_table_set")
    return value


def _table_name_for_set(table_set: str) -> str:
    return TABLE_TEST if table_set == "test" else TABLE_PROD


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


def _get_store(config: dict, table_set: str = "prod") -> ExperimentStore:
    global _store, _store_key, _stores
    table_set = _parse_table_set(table_set)
    exp_cfg = config.get("experiments", {})
    cache_key = _store_cache_key(exp_cfg) + ":" + table_set
    if cache_key not in _stores:
        try:
            _stores[cache_key] = create_experiment_store(
                exp_cfg, ROOT, table_name=_table_name_for_set(table_set)
            )
        except RuntimeError as exc:
            if str(exc) == "pymysql_not_installed":
                raise HTTPException(status_code=500, detail="pymysql_not_installed") from exc
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"database_unavailable: {exc}") from exc
    store = _stores[cache_key]
    if table_set == "prod":
        _store = store
        _store_key = cache_key
    return store


def _handle_value_error(exc: ValueError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def _debug_output_root(config: dict) -> Path:
    return ROOT / config.get("debug", {}).get("output_dir", "debug_output")


def _debug_api_key(config: dict) -> str:
    return config.get("debug", {}).get("api_key", "iris-color-dev")


def _experiment_snapshot_root(config: dict, table_set: str = "prod") -> Path:
    exp_cfg = config.get("experiments", {})
    if _parse_table_set(table_set) == "test":
        rel = exp_cfg.get("snapshot_dir_test", "data/experiment_snapshots_test")
    else:
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


def _persist_compare_snapshots(
    config: dict, store: ExperimentStore, record: dict, *, table_set: str = "prod"
) -> dict:
    """保存实验记录后，持久化调色前后对比图（debug run 或原图重识别）。"""
    if record.get("image_before_rel") and record.get("image_after_rel"):
        return record

    record_id = int(record["id"])
    snapshot_root = _experiment_snapshot_root(config, table_set)

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
    *,
    table_set: str = "prod",
) -> dict:
    """确保记录已有调色前后快照；若无则按 debug run 或 image_rel 生成。"""
    record = store.get_by_id(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="record_not_found")

    snapshot_root = _experiment_snapshot_root(config, table_set)
    try:
        if record.get("image_before_rel") and record.get("image_after_rel"):
            out = dict(record)
        else:
            out = _persist_compare_snapshots(config, store, record, table_set=table_set)
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

    api_key = config.get("experiments", {}).get("api_key", "iris-color-dev")
    urls = experiment_snapshot_urls(record_id, api_key, table_set=table_set)
    snap_prefix = "test/snapshots" if table_set == "test" else "snapshots"
    result = {
        **out,
        "thumb_before_url": urls["thumb_before_url"],
        "thumb_after_url": urls["thumb_after_url"],
        "compare_audit": audit,
    }
    if audit:
        result["audit_url"] = f"/experiments/{snap_prefix}/{record_id}/{COMPARE_AUDIT_FILENAME}?key={api_key}"
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
    table_set: str = Query("prod", description="prod=正式记录, test=测试记录"),
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
    ts = _parse_table_set(table_set)
    store = _get_store(config, ts)
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
    return {"records": records, "count": len(records), "table_set": ts}


@router.get("/records/{record_id}")
def get_record(
    record_id: int,
    key: str = Query(...),
    table_set: str = Query("prod"),
):
    config = _verify_key(key)
    ts = _parse_table_set(table_set)
    store = _get_store(config, ts)
    record = store.get_by_id(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="record_not_found")
    return record


@router.post("/records/{record_id}/ensure-compare-images")
def ensure_compare_images(
    record_id: int,
    key: str = Query(...),
    table_set: str = Query("prod"),
    force: bool = Query(False, description="强制用当前算法重算并覆盖已有快照"),
):
    """按需生成或返回实验记录的调色前后对比图。"""
    config = _verify_key(key)
    ts = _parse_table_set(table_set)
    store = _get_store(config, ts)
    api_key = config.get("experiments", {}).get("api_key", "iris-color-dev")
    snap_prefix = "test/snapshots" if ts == "test" else "snapshots"
    try:
        if force:
            record = store.get_by_id(record_id)
            if not record:
                raise HTTPException(status_code=404, detail="record_not_found")
            snapshot_root = _experiment_snapshot_root(config, ts)
            from app.services.experiment_debug_import import generate_compare_snapshots_for_record

            paths = generate_compare_snapshots_for_record(
                record,
                snapshot_root,
                record_id,
                config,
                CONFIG_PATH,
                IMG_ROOT,
                _debug_output_root(config),
                force=True,
            )
            if paths.get("image_before_rel"):
                record = _apply_compare_snapshot_paths(store, record, paths)
            out = dict(record)
            audit = load_compare_audit(snapshot_root, record_id)
            if audit:
                audit = enrich_audit_with_current_standards(audit, config)
            urls = experiment_snapshot_urls(record_id, api_key, table_set=ts)
            return {
                **out,
                "thumb_before_url": urls["thumb_before_url"],
                "thumb_after_url": urls["thumb_after_url"],
                "compare_audit": audit,
                "audit_url": f"/experiments/{snap_prefix}/{record_id}/{COMPARE_AUDIT_FILENAME}?key={api_key}"
                if audit
                else None,
            }
        return ensure_record_compare_images(config, store, record_id, table_set=ts)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.post("/records/{record_id}/recognize")
def recognize_record_preview(
    record_id: int,
    key: str = Query(...),
    table_set: str = Query("prod"),
    body: dict = Body(default={}),
):
    """对关联原图用当前算法识别（预览，不写库）。"""
    config = _verify_key(key)
    ts = _parse_table_set(table_set)
    store = _get_store(config, ts)
    record = store.get_by_id(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="record_not_found")
    try:
        return build_record_recognition_preview(
            record,
            config,
            CONFIG_PATH,
            IMG_ROOT,
            _debug_output_root(config),
            skip_quality=body.get("skip_quality"),
            manual_params=body.get("manual_params"),
            closeup_mode=body.get("mode") or "auto",
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.post("/records/{record_id}/recognize/apply")
def recognize_record_apply(
    record_id: int,
    key: str = Query(...),
    table_set: str = Query("prod"),
    body: dict = Body(...),
):
    """应用重新识别结果，可选覆盖实验记录字段与对比快照。"""
    config = _verify_key(key)
    ts = _parse_table_set(table_set)
    store = _get_store(config, ts)
    record = store.get_by_id(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="record_not_found")

    overwrite_record = bool(body.get("overwrite_record", False))
    overwrite_snapshots = bool(body.get("overwrite_snapshots", False))
    if not overwrite_record and not overwrite_snapshots:
        raise HTTPException(status_code=400, detail="nothing_to_overwrite")

    try:
        result = apply_record_recognition(
            record,
            store,
            config,
            CONFIG_PATH,
            IMG_ROOT,
            _debug_output_root(config),
            _experiment_snapshot_root(config, ts),
            overwrite_record=overwrite_record,
            overwrite_snapshots=overwrite_snapshots,
            skip_quality=body.get("skip_quality"),
            manual_params=body.get("manual_params"),
            closeup_mode=body.get("mode") or "auto",
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc

    api_key = config.get("experiments", {}).get("api_key", "iris-color-dev")
    out_record = result["record"]
    if out_record.get("image_before_rel") and out_record.get("image_after_rel"):
        urls = experiment_snapshot_urls(record_id, api_key, table_set=ts)
        out_record = {
            **out_record,
            "thumb_before_url": urls["thumb_before_url"],
            "thumb_after_url": urls["thumb_after_url"],
        }
    return result


@router.post("/records", status_code=201)
def create_record(
    key: str = Query(...),
    table_set: str = Query("prod"),
    payload: dict = Body(...),
):
    config = _verify_key(key)
    ts = _parse_table_set(table_set)
    store = _get_store(config, ts)
    try:
        record = store.create_record(payload)
        record = _persist_compare_snapshots(config, store, record, table_set=ts)
        return record
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.put("/records/{record_id}")
def update_record(
    record_id: int,
    key: str = Query(...),
    table_set: str = Query("prod"),
    payload: dict = Body(...),
):
    config = _verify_key(key)
    ts = _parse_table_set(table_set)
    store = _get_store(config, ts)
    try:
        record = store.update_record(record_id, payload)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    if not record:
        raise HTTPException(status_code=404, detail="record_not_found")
    record = _persist_compare_snapshots(config, store, record, table_set=ts)
    return record


@router.patch("/records/{record_id}/include-in-stats")
def patch_include_in_stats(
    record_id: int,
    key: str = Query(...),
    table_set: str = Query("prod"),
    body: dict = Body(...),
):
    """快速切换是否纳入统计（表格内勾选）。"""
    config = _verify_key(key)
    ts = _parse_table_set(table_set)
    store = _get_store(config, ts)
    if "include_in_stats" not in body:
        raise HTTPException(status_code=400, detail="missing_include_in_stats")
    record = store.update_include_in_stats(record_id, bool(body["include_in_stats"]))
    if not record:
        raise HTTPException(status_code=404, detail="record_not_found")
    return record


@router.delete("/records/{record_id}")
def delete_record(
    record_id: int,
    key: str = Query(...),
    table_set: str = Query("prod"),
):
    config = _verify_key(key)
    ts = _parse_table_set(table_set)
    store = _get_store(config, ts)
    if not store.delete_record(record_id):
        raise HTTPException(status_code=404, detail="record_not_found")
    return {"ok": True, "id": record_id}


@router.post("/records/bulk-delete")
def bulk_delete(
    key: str = Query(...),
    table_set: str = Query("prod"),
    payload: dict = Body(...),
):
    config = _verify_key(key)
    ts = _parse_table_set(table_set)
    store = _get_store(config, ts)
    ids = payload.get("ids") or []
    if not isinstance(ids, list):
        raise HTTPException(status_code=400, detail="invalid_ids")
    try:
        id_list = [int(i) for i in ids]
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="invalid_ids") from exc
    deleted = store.bulk_delete(id_list)
    return {"ok": True, "deleted": deleted}


@router.post("/records/copy-to-test")
def copy_records_to_test(key: str = Query(...), payload: dict = Body(default={})):
    """从正式记录表复制到测试记录表（不含快照，重识别时会重新生成）。"""
    config = _verify_key(key)
    prod_store = _get_store(config, "prod")
    test_store = _get_store(config, "test")
    ids = payload.get("ids")
    if ids is not None:
        if not isinstance(ids, list):
            raise HTTPException(status_code=400, detail="invalid_ids")
        try:
            ids = [int(i) for i in ids]
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="invalid_ids") from exc
        if not ids:
            raise HTTPException(status_code=400, detail="empty_ids")
    result = test_store.copy_records_from(prod_store, ids)
    return {"ok": True, **result}


@router.post("/records/bulk-recognize")
def bulk_recognize_records(
    key: str = Query(...),
    table_set: str = Query("test", description="默认 test；可对正式/测试表批量重识别"),
    payload: dict = Body(...),
):
    """批量重识别：对选中记录用当前算法重新识别并写回。"""
    config = _verify_key(key)
    ts = _parse_table_set(table_set)
    store = _get_store(config, ts)
    ids = payload.get("ids") or []
    if not isinstance(ids, list):
        raise HTTPException(status_code=400, detail="invalid_ids")
    try:
        id_list = [int(i) for i in ids]
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="invalid_ids") from exc
    if not id_list:
        raise HTTPException(status_code=400, detail="empty_ids")

    overwrite_record = bool(payload.get("overwrite_record", True))
    overwrite_snapshots = bool(payload.get("overwrite_snapshots", True))
    if not overwrite_record and not overwrite_snapshots:
        raise HTTPException(status_code=400, detail="nothing_to_overwrite")

    try:
        return bulk_apply_record_recognition(
            id_list,
            store,
            config,
            CONFIG_PATH,
            IMG_ROOT,
            _debug_output_root(config),
            _experiment_snapshot_root(config, ts),
            overwrite_record=overwrite_record,
            overwrite_snapshots=overwrite_snapshots,
            skip_quality=payload.get("skip_quality"),
            closeup_mode=payload.get("mode") or "auto",
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.get("/meta/options")
def meta_options(key: str = Query(...), table_set: str = Query("prod")):
    config = _verify_key(key)
    ts = _parse_table_set(table_set)
    store = _get_store(config, ts)
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
    """读取实验记录持久化的调色前/后快照或审计 JSON（正式表）。"""
    return _get_experiment_snapshot(record_id, filename, key, table_set="prod")


@router.get("/test/snapshots/{record_id}/{filename}")
def get_test_experiment_snapshot(
    record_id: int,
    filename: str,
    key: str = Query(...),
):
    """读取测试记录表持久化的调色前/后快照或审计 JSON。"""
    return _get_experiment_snapshot(record_id, filename, key, table_set="test")


def _get_experiment_snapshot(record_id: int, filename: str, key: str, *, table_set: str):
    config = _verify_key(key)
    ts = _parse_table_set(table_set)
    allowed = ("before.jpg", "after.jpg", COMPARE_AUDIT_FILENAME)
    if filename not in allowed:
        raise HTTPException(status_code=400, detail="invalid_snapshot_file")
    path = _experiment_snapshot_root(config, ts) / str(record_id) / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="snapshot_not_found")
    if filename == COMPARE_AUDIT_FILENAME:
        audit = load_compare_audit(_experiment_snapshot_root(config, ts), record_id)
        if not audit:
            raise HTTPException(status_code=404, detail="audit_not_found")
        return enrich_audit_with_current_standards(audit, config)
    return FileResponse(path, media_type="image/jpeg")


def _grade_boundaries(config: dict) -> list[float]:
    return (config.get("grade") or {}).get("boundaries", [55, 45, 29, 19])


@router.post("/stats/stability")
def stability_stats(
    key: str = Query(...),
    table_set: str = Query("prod"),
    body: dict = Body(default={}),
):
    """L* 离散度稳定性分析（支持范围筛选，供管理页图表使用）。"""
    config = _verify_key(key)
    ts = _parse_table_set(table_set)
    store = _get_store(config, ts)
    records = store.list_records()
    boundaries = _grade_boundaries(config)

    min_subgroup_n = int(body.get("min_subgroup_n") or 3)
    min_subgroup_n = max(2, min(min_subgroup_n, 50))

    operators = body.get("operators")
    if operators is not None and not isinstance(operators, list):
        raise HTTPException(status_code=400, detail="invalid_operators")
    group_names = body.get("group_names")
    if group_names is not None and not isinstance(group_names, list):
        raise HTTPException(status_code=400, detail="invalid_group_names")
    subgroup_names = body.get("subgroup_names")
    if subgroup_names is not None and not isinstance(subgroup_names, list):
        raise HTTPException(status_code=400, detail="invalid_subgroup_names")
    record_ids = body.get("record_ids")
    if record_ids is not None:
        if not isinstance(record_ids, list):
            raise HTTPException(status_code=400, detail="invalid_record_ids")
        try:
            record_ids = [int(i) for i in record_ids]
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="invalid_record_ids") from exc

    # 与列表页一致的字段筛选（可选）
    field_filters = {
        "group_name": body.get("group_name"),
        "subgroup_name": body.get("subgroup_name"),
        "date_from": body.get("date_from"),
        "date_to": body.get("date_to"),
        "operator": body.get("operator"),
        "camera_device": body.get("camera_device"),
        "light_device": body.get("light_device"),
        "color": body.get("color"),
        "grade_before": body.get("grade_before"),
        "grade_after": body.get("grade_after"),
    }
    filtered = records
    for field, val in field_filters.items():
        if val is None or val == "":
            continue
        if field in ("date_from", "date_to"):
            if field == "date_from":
                filtered = [r for r in filtered if (r.get("experiment_date") or "") >= str(val)]
            else:
                filtered = [r for r in filtered if (r.get("experiment_date") or "") <= str(val)]
        else:
            filtered = [r for r in filtered if r.get(field) == val]

    report = analyze_stability(
        filtered,
        boundaries,
        min_subgroup_n=min_subgroup_n,
        operators=operators or None,
        group_names=group_names or None,
        subgroup_names=subgroup_names or None,
        record_ids=record_ids or None,
    )
    report["source_count"] = len(records)
    report["filtered_count"] = len(filtered)
    report["table_set"] = ts
    return report
