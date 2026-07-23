"""调试后门 API（不对接前端业务）。"""

import json
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Body, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from app.models.schemas import DebugAnalysisResponse
from app.services.debug_viz import (
    build_debug_images,
    build_debug_metrics,
    images_to_base64,
    save_debug_run,
)
from app.services.pipeline import AnalysisError, load_config, run_analysis

router = APIRouter(prefix="/debug", tags=["debug"])

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "config" / "grade_thresholds.yaml"
DEFAULT_OUTPUT = ROOT / "debug_output"
IMG_ROOT = ROOT.parent / "img"
IMG_EXTS = (".jpg", ".jpeg", ".png")
LABELS_PATH = ROOT / "labels" / "img_labels.json"
# 后缀：原名__G2_L51.2_M.jpg（保留原名排序）；兼容旧前缀 G2_L51.2_M_原名.jpg
LABEL_SUFFIX_RE = re.compile(r"__G([1-5])_L([\d.]+)(_M)?$", re.IGNORECASE)
LABEL_SUFFIX_GRADE_ONLY_RE = re.compile(r"__G([1-5])(_M)?$", re.IGNORECASE)
LABEL_PREFIX_RE = re.compile(r"^G([1-5])_L([\d.]+)_", re.IGNORECASE)
LEGACY_GRADE_PREFIX_RE = re.compile(r"^G[1-5]_", re.IGNORECASE)


def _verify_debug_key(provided: Optional[str], config: dict) -> None:
    debug_cfg = config.get("debug", {})
    if not debug_cfg.get("enabled", True):
        raise HTTPException(status_code=403, detail="debug_disabled")
    expected = debug_cfg.get("api_key", "iris-color-dev")
    if provided != expected:
        raise HTTPException(status_code=403, detail="invalid_debug_key")


def _safe_img_rel(rel: str) -> Path:
    """把前端传来的相对路径限制在 img/ 目录内，避免路径穿越。"""
    rel = (rel or "").replace("\\", "/").lstrip("/")
    target = (IMG_ROOT / rel).resolve()
    if not str(target).startswith(str(IMG_ROOT.resolve())):
        raise HTTPException(status_code=400, detail="invalid_path")
    return target


def _list_img_files() -> list[str]:
    if not IMG_ROOT.exists():
        return []
    items = []
    for path in IMG_ROOT.rglob("*"):
        if path.suffix.lower() in IMG_EXTS and path.is_file():
            items.append(path.relative_to(IMG_ROOT).as_posix())
    return sorted(items)


def _strip_label_tags(stem: str) -> str:
    """去掉文件名中的 Grade/L*/人工调整标记（后缀或旧前缀）。"""
    for pattern in (LABEL_SUFFIX_RE, LABEL_SUFFIX_GRADE_ONLY_RE):
        match = pattern.search(stem)
        if match:
            return stem[: match.start()]
    match = LABEL_PREFIX_RE.match(stem)
    if match:
        rest = stem[match.end() :]
        if len(rest) >= 2 and rest[0] in "Mm" and rest[1] == "_":
            rest = rest[2:]
        return rest
    legacy = LEGACY_GRADE_PREFIX_RE.match(stem)
    if legacy:
        rest = stem[legacy.end() :]
        if len(rest) >= 2 and rest[0] in "Mm" and rest[1] == "_":
            rest = rest[2:]
        return rest
    return stem


def _manual_marker_after_l_star(stem: str, prefix_end: int) -> bool:
    rest = stem[prefix_end:]
    return len(rest) >= 2 and rest[0] in "Mm" and rest[1] == "_"


def _format_l_star(l_star: float) -> str:
    return f"{l_star:.1f}"


def _format_label_tag(grade: int, l_star: Optional[float], *, manual_adjusted: bool) -> str:
    manual_part = "_M" if manual_adjusted else ""
    if l_star is not None:
        return f"__G{grade}_L{_format_l_star(l_star)}{manual_part}"
    if manual_adjusted:
        return f"__G{grade}_M"
    return f"__G{grade}"


def _parse_label_tags(filename: str) -> dict:
    stem = Path(filename).stem
    suffix = LABEL_SUFFIX_RE.search(stem)
    if suffix:
        return {
            "grade_prefix": int(suffix.group(1)),
            "l_star_prefix": float(suffix.group(2)),
            "manual_adjusted": bool(suffix.group(3)),
        }
    suffix_grade = LABEL_SUFFIX_GRADE_ONLY_RE.search(stem)
    if suffix_grade:
        return {
            "grade_prefix": int(suffix_grade.group(1)),
            "l_star_prefix": None,
            "manual_adjusted": bool(suffix_grade.group(2)),
        }
    match = LABEL_PREFIX_RE.match(stem)
    if match:
        return {
            "grade_prefix": int(match.group(1)),
            "l_star_prefix": float(match.group(2)),
            "manual_adjusted": _manual_marker_after_l_star(stem, match.end()),
        }
    legacy = LEGACY_GRADE_PREFIX_RE.match(stem)
    if legacy:
        return {
            "grade_prefix": int(legacy.group(0)[1]),
            "l_star_prefix": None,
            "manual_adjusted": _manual_marker_after_l_star(stem, legacy.end()),
        }
    return {"grade_prefix": None, "l_star_prefix": None, "manual_adjusted": False}


def _apply_label_tags(
    filename: str,
    grade: int,
    l_star: Optional[float] = None,
    *,
    manual_adjusted: bool = False,
) -> str:
    path = Path(filename)
    base = _strip_label_tags(path.stem)
    tag = _format_label_tag(grade, l_star, manual_adjusted=manual_adjusted)
    return f"{base}{tag}{path.suffix}"


_strip_label_prefix = _strip_label_tags
_parse_label_prefix = _parse_label_tags
_apply_label_prefix = _apply_label_tags


def _load_img_labels() -> dict:
    if LABELS_PATH.exists():
        try:
            return json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_img_labels(data: dict) -> None:
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _upsert_img_label(
    labels: dict,
    *,
    new_rel: str,
    old_rel: str,
    grade: int,
    lab: Optional[dict] = None,
    iris_color: Optional[dict] = None,
    confidence: Optional[float] = None,
    manual_adjusted: bool = False,
) -> None:
    from datetime import datetime, timezone

    if old_rel in labels and old_rel != new_rel:
        del labels[old_rel]
    labels[new_rel] = {
        "grade": grade,
        "lab": lab,
        "iris_color": iris_color,
        "confidence": confidence,
        "manual_adjusted": manual_adjusted,
        "previous_rel": old_rel if old_rel != new_rel else labels.get(new_rel, {}).get("previous_rel"),
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _decode_image(content: bytes):
    arr = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="invalid_image_format")
    return image


def _parse_manual_params(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="invalid_manual_params_json") from exc
    required = ["center_x", "center_y", "pupil_radius", "inner_radius", "outer_radius"]
    if not isinstance(data, dict) or any(key not in data for key in required):
        raise HTTPException(status_code=400, detail="manual_params_missing_fields")
    try:
        return {key: float(data[key]) for key in required}
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="manual_params_must_be_numbers") from exc


def _build_debug_response(
    image_bgr,
    config: dict,
    pipeline,
    *,
    include_base64: bool,
    source_rel: Optional[str] = None,
    source_filename: Optional[str] = None,
) -> DebugAnalysisResponse:
    debug_cfg = config.get("debug", {})
    highlight_v = config.get("highlight_v_threshold", 240)
    eye_cfg = config.get("eye_closeup", {})
    images = build_debug_images(
        pipeline,
        eye_cfg,
        config=config,
        highlight_v=highlight_v,
    )
    # 00_original 保留原始分辨率：调试台人工校准画布依赖原图坐标系
    all_images = {"00_original": image_bgr, **images}
    metrics = build_debug_metrics(pipeline, highlight_v, config=config)
    if source_rel:
        metrics["source_rel"] = source_rel
        metrics["image_rel"] = source_rel
    if source_filename:
        metrics["source_filename"] = source_filename
        metrics["original_filename"] = source_filename

    run_id = None
    saved_dir = None
    image_urls = {}

    if debug_cfg.get("save_to_disk", True):
        out_root = ROOT / debug_cfg.get("output_dir", "debug_output")
        run_id, run_dir = save_debug_run(out_root, image_bgr, images, metrics)
        saved_dir = str(run_dir.relative_to(ROOT)).replace("\\", "/")

    api_key = debug_cfg.get("api_key", "iris-color-dev")
    if run_id:
        image_urls = {
            name: f"/debug/files/{run_id}/{name}.jpg?key={api_key}"
            for name in all_images.keys()
        }
        image_urls["metrics"] = f"/debug/files/{run_id}/metrics.json?key={api_key}"

    viewer_url = f"/debug/viewer/{run_id}?key={api_key}" if run_id else None

    message = (
        "debug run saved; open viewer_url in browser"
        if run_id
        else "debug run generated without disk output"
    )

    return DebugAnalysisResponse(
        success=True,
        run_id=run_id,
        saved_dir=saved_dir,
        viewer_url=viewer_url,
        metrics=metrics,
        image_urls=image_urls,
        images_base64=images_to_base64(all_images) if include_base64 or not run_id else None,
        message=message,
    )


@router.post("/analyze", response_model=DebugAnalysisResponse)
async def analyze_debug(
    file: UploadFile = File(...),
    x_debug_key: Optional[str] = Header(None, alias="X-Debug-Key"),
    key: Optional[str] = Query(None, description="与 X-Debug-Key 二选一，便于调试页表单提交"),
    skip_quality: bool = Query(False, description="跳过模糊/过曝质量门槛，便于测试糊图"),
    include_base64: bool = Query(False, description="是否在 JSON 内返回 base64 图片（体积大）"),
    mode: str = Query("auto", description="眼部特写识别模式：auto/precise/rough"),
    source_rel: Optional[str] = Form(None, description="img/ 内相对路径，供实验记录关联"),
):
    """
    调试专用分析接口。

    - 请求头 `X-Debug-Key` 或 query `key`（调试页用）
    - 保存可视化到 `debug_output/{run_id}/`
    - 浏览器打开 `/debug/ui?key=...` 可拍照/上传并直接看结果
    """
    config = load_config(CONFIG_PATH)
    _verify_debug_key(x_debug_key or key, config)

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty_file")

    image_bgr = _decode_image(content)
    closeup_mode = mode if mode in ("auto", "precise", "rough") else "auto"
    try:
        pipeline = run_analysis(
            image_bgr,
            config,
            CONFIG_PATH,
            skip_quality=skip_quality,
            closeup_mode=closeup_mode,
        )
    except AnalysisError as exc:
        detail = {"success": False, "error": exc.code}
        if exc.quality:
            detail["quality"] = {
                "blur_score": exc.quality.blur_score,
                "overexposed_ratio": exc.quality.overexposed_ratio,
                "eye_open": exc.quality.eye_open,
                "issues": exc.quality.issues,
            }
        raise HTTPException(status_code=422, detail=detail) from exc

    return _build_debug_response(
        image_bgr, config, pipeline, include_base64=include_base64,
        source_rel=source_rel, source_filename=file.filename,
    )


@router.post("/analyze/manual", response_model=DebugAnalysisResponse)
async def analyze_debug_manual(
    file: UploadFile = File(...),
    manual_params: str = Form(..., description="JSON: center_x, center_y, pupil_radius, inner_radius, outer_radius"),
    x_debug_key: Optional[str] = Header(None, alias="X-Debug-Key"),
    key: Optional[str] = Query(None, description="与 X-Debug-Key 二选一，便于调试页表单提交"),
    skip_quality: bool = Query(False, description="跳过模糊/过曝质量门槛，便于测试糊图"),
    include_base64: bool = Query(False, description="是否在 JSON 内返回 base64 图片（体积大）"),
    source_rel: Optional[str] = Form(None, description="img/ 内相对路径，供实验记录关联"),
):
    """调试专用：按人工调整的瞳孔/环带参数重新分析并保存新 run。"""
    config = load_config(CONFIG_PATH)
    _verify_debug_key(x_debug_key or key, config)

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty_file")

    image_bgr = _decode_image(content)
    manual_detection = _parse_manual_params(manual_params)

    try:
        pipeline = run_analysis(
            image_bgr,
            config,
            CONFIG_PATH,
            skip_quality=skip_quality,
            manual_detection=manual_detection,
        )
    except AnalysisError as exc:
        detail = {"success": False, "error": exc.code}
        if exc.quality:
            detail["quality"] = {
                "blur_score": exc.quality.blur_score,
                "overexposed_ratio": exc.quality.overexposed_ratio,
                "eye_open": exc.quality.eye_open,
                "issues": exc.quality.issues,
            }
        raise HTTPException(status_code=422, detail=detail) from exc

    return _build_debug_response(
        image_bgr, config, pipeline, include_base64=include_base64,
        source_rel=source_rel, source_filename=file.filename,
    )


@router.get("/viewer/{run_id}", response_class=HTMLResponse)
async def debug_viewer(
    run_id: str,
    key: str = Query(..., alias="key"),
):
    """本地 HTML 查看某次调试运行的各阶段图片（非前端业务页）。"""
    config = load_config(CONFIG_PATH)
    if key != config.get("debug", {}).get("api_key", "iris-color-dev"):
        raise HTTPException(status_code=403, detail="invalid_debug_key")

    run_dir = ROOT / config.get("debug", {}).get("output_dir", "debug_output") / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="run_not_found")

    panels = [
        ("00_original", "原图"),
        ("01_pupil_candidates", "瞳孔候选（黄=极暗候选区域）"),
        ("01_pupil_localization", "瞳孔定位（黄框=搜索区，蓝圆=瞳孔，红十字=中心）"),
        ("02_iris_ring", "虹膜环带（绿=外缘，红=内缘，青=采样环）"),
        ("03_highlight_rejection", "剔除干扰（红=环内被剔除的高光/过亮/极暗像素，外围变暗）"),
        ("04_valid_samples", "最终取色像素（绿色=参与 Lab 中位数）"),
        ("05_ring_mask_only", "环带 mask 伪彩色"),
        ("06_sclera_samples", "巩膜参考采样（品红=采样像素，黄圆=环带）"),
        ("07_sclera_before_after", "巩膜调色前 / 调色后对比（裁切 + Lab 色块）"),
        ("08_iris_before", "调色前虹膜裁切"),
        ("09_iris_after", "调色后虹膜裁切"),
    ]
    items = ""
    for key_name, title in panels:
        img_path = run_dir / f"{key_name}.jpg"
        if not img_path.exists():
            continue
        src = f"/debug/files/{run_id}/{key_name}.jpg?key={key}"
        items += f"""
        <section>
          <h3>{title}</h3>
          <img src="{src}" alt="{key_name}" />
        </section>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="UTF-8"/><title>Debug {run_id}</title>
<style>
body {{ font-family: sans-serif; background:#111; color:#eee; margin:16px; }}
section {{ margin-bottom:24px; border:1px solid #333; padding:12px; border-radius:8px; }}
img {{ max-width:100%; height:auto; background:#000; }}
h1 {{ font-size:20px; }}
h3 {{ margin:0 0 8px; color:#9cf; }}
a {{ color:#6cf; }}
</style></head><body>
<h1>iris-vision 调试查看 — {run_id}</h1>
<p><a href="/debug/files/{run_id}/metrics.json?key={key}" target="_blank">metrics.json</a></p>
{items}
</body></html>"""
    return HTMLResponse(html)


@router.get("/img/list")
def list_img_files(
    x_debug_key: Optional[str] = Header(None, alias="X-Debug-Key"),
    key: Optional[str] = Query(None),
):
    """列出 img/ 下全部图片（含子文件夹），供调试台载入与重命名。"""
    _verify_debug_key(x_debug_key or key, load_config(CONFIG_PATH))
    images = []
    for rel in _list_img_files():
        name = Path(rel).name
        parsed = _parse_label_prefix(name)
        images.append(
            {
                "rel": rel,
                "name": name,
                **parsed,
            }
        )
    return {"images": images, "count": len(images)}


@router.get("/img/file")
def get_img_file(
    rel: str = Query(...),
    x_debug_key: Optional[str] = Header(None, alias="X-Debug-Key"),
    key: Optional[str] = Query(None),
):
    """读取 img/ 内图片原文件，供调试台载入分析。"""
    _verify_debug_key(x_debug_key or key, load_config(CONFIG_PATH))
    path = _safe_img_rel(rel)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="image_not_found")
    return FileResponse(path)


@router.post("/rename-grade")
def rename_with_grade_prefix(
    payload: dict = Body(...),
    x_debug_key: Optional[str] = Header(None, alias="X-Debug-Key"),
    key: Optional[str] = Query(None),
):
    """将 img/ 内图片重命名为 原名__G{n}_L{L*}_ 后缀，并写入 labels/img_labels.json。"""
    _verify_debug_key(x_debug_key or key, load_config(CONFIG_PATH))
    rel = payload.get("rel")
    grade = payload.get("grade")
    lab = payload.get("lab")
    iris_color = payload.get("iris_color")
    confidence = payload.get("confidence")
    l_star = payload.get("l_star")
    manual_adjusted = bool(payload.get("manual_adjusted", False))
    if lab and isinstance(lab, dict) and "L" in lab:
        l_star = lab["L"]
    if not rel:
        raise HTTPException(status_code=400, detail="missing_rel")
    try:
        grade = int(grade)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="invalid_grade") from exc
    if grade not in (1, 2, 3, 4, 5):
        raise HTTPException(status_code=400, detail="grade_out_of_range")
    if l_star is not None:
        try:
            l_star = float(l_star)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="invalid_l_star") from exc

    path = _safe_img_rel(rel)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="image_not_found")

    new_name = _apply_label_prefix(path.name, grade, l_star, manual_adjusted=manual_adjusted)
    if new_name == path.name:
        labels = _load_img_labels()
        _upsert_img_label(
            labels,
            new_rel=rel,
            old_rel=rel,
            grade=grade,
            lab=lab if isinstance(lab, dict) else None,
            iris_color=iris_color if isinstance(iris_color, dict) else None,
            confidence=float(confidence) if confidence is not None else None,
            manual_adjusted=manual_adjusted,
        )
        _save_img_labels(labels)
        return {
            "ok": True,
            "rel": rel,
            "new_rel": rel,
            "new_name": new_name,
            "grade": grade,
            "l_star": l_star,
            "manual_adjusted": manual_adjusted,
            "labels_path": str(LABELS_PATH.relative_to(ROOT)).replace("\\", "/"),
            "message": "filename_unchanged",
        }

    new_path = path.with_name(new_name)
    if new_path.exists():
        raise HTTPException(status_code=409, detail="target_exists")

    path.rename(new_path)
    new_rel = new_path.relative_to(IMG_ROOT).as_posix()
    labels = _load_img_labels()
    _upsert_img_label(
        labels,
        new_rel=new_rel,
        old_rel=rel,
        grade=grade,
        lab=lab if isinstance(lab, dict) else None,
        iris_color=iris_color if isinstance(iris_color, dict) else None,
        confidence=float(confidence) if confidence is not None else None,
        manual_adjusted=manual_adjusted,
    )
    _save_img_labels(labels)
    return {
        "ok": True,
        "rel": rel,
        "new_rel": new_rel,
        "new_name": new_name,
        "grade": grade,
        "l_star": l_star,
        "manual_adjusted": manual_adjusted,
        "labels_path": str(LABELS_PATH.relative_to(ROOT)).replace("\\", "/"),
    }


@router.get("/ui", response_class=HTMLResponse)
async def debug_ui():
    """调试台：浏览器内拍照/上传 + 查看各阶段图（推荐入口）。"""
    config = load_config(CONFIG_PATH)
    if not config.get("debug", {}).get("enabled", True):
        raise HTTPException(status_code=403, detail="debug_disabled")
    html_path = ROOT / "app" / "static" / "debug_ui.html"
    if not html_path.exists():
        raise HTTPException(status_code=500, detail="debug_ui_missing")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@router.get("", response_class=HTMLResponse)
async def debug_root():
    """重定向到调试台。"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/debug/ui")


@router.get("/runs")
async def list_debug_runs(
    x_debug_key: Optional[str] = Header(None, alias="X-Debug-Key"),
    key: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
):
    """列出最近调试运行目录。"""
    config = load_config(CONFIG_PATH)
    _verify_debug_key(x_debug_key or key, config)
    out_root = ROOT / config.get("debug", {}).get("output_dir", "debug_output")
    if not out_root.exists():
        return {"runs": []}
    runs = sorted([p.name for p in out_root.iterdir() if p.is_dir()], reverse=True)[:limit]
    return {
        "runs": [
            {
                "run_id": r,
                "viewer_url": f"/debug/viewer/{r}?key={config.get('debug', {}).get('api_key', 'iris-color-dev')}",
            }
            for r in runs
        ]
    }
