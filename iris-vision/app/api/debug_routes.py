"""调试后门 API（不对接前端业务）。"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse

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


def _verify_debug_key(provided: Optional[str], config: dict) -> None:
    debug_cfg = config.get("debug", {})
    if not debug_cfg.get("enabled", True):
        raise HTTPException(status_code=403, detail="debug_disabled")
    expected = debug_cfg.get("api_key", "iris-color-dev")
    if provided != expected:
        raise HTTPException(status_code=403, detail="invalid_debug_key")


def _decode_image(content: bytes):
    import cv2
    import numpy as np

    arr = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="invalid_image_format")
    return image


@router.post("/analyze", response_model=DebugAnalysisResponse)
async def analyze_debug(
    file: UploadFile = File(...),
    x_debug_key: Optional[str] = Header(None, alias="X-Debug-Key"),
    key: Optional[str] = Query(None, description="与 X-Debug-Key 二选一，便于调试页表单提交"),
    skip_quality: bool = Query(False, description="跳过模糊/过曝质量门槛，便于测试糊图"),
    include_base64: bool = Query(False, description="是否在 JSON 内返回 base64 图片（体积大）"),
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
    debug_cfg = config.get("debug", {})
    highlight_v = config.get("highlight_v_threshold", 240)
    eye_cfg = config.get("eye_closeup", {})

    try:
        pipeline = run_analysis(
            image_bgr,
            config,
            CONFIG_PATH,
            skip_quality=skip_quality,
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

    images = build_debug_images(image_bgr, pipeline, eye_cfg)
    metrics = build_debug_metrics(pipeline, highlight_v)

    run_id = None
    saved_dir = None
    image_urls = {}

    if debug_cfg.get("save_to_disk", True):
        out_root = ROOT / debug_cfg.get("output_dir", "debug_output")
        run_id, run_dir = save_debug_run(out_root, image_bgr, images, metrics)
        saved_dir = str(run_dir.relative_to(ROOT)).replace("\\", "/")

    api_key = debug_cfg.get("api_key", "iris-color-dev")
    image_urls = {}
    if run_id:
        image_urls = {
            name: f"/debug/files/{run_id}/{name}.jpg?key={api_key}"
            for name in ["00_original"] + list(images.keys())
        }
        image_urls["metrics"] = f"/debug/files/{run_id}/metrics.json?key={api_key}"

    viewer_url = f"/debug/viewer/{run_id}?key={api_key}" if run_id else None

    return DebugAnalysisResponse(
        success=True,
        run_id=run_id,
        saved_dir=saved_dir,
        viewer_url=viewer_url,
        metrics=metrics,
        image_urls=image_urls,
        images_base64=images_to_base64(images) if include_base64 else None,
        message="debug run saved; open viewer_url in browser",
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
    ]
    items = ""
    for key_name, title in panels:
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
