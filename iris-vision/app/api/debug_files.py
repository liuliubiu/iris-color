"""调试静态文件与 viewer 密钥校验。"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

router = APIRouter(prefix="/debug/files", tags=["debug"])

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "config" / "grade_thresholds.yaml"


def _expected_key() -> str:
    import yaml

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("debug", {}).get("api_key", "iris-color-dev")


def _verify_key(key: Optional[str]) -> None:
    if key != _expected_key():
        raise HTTPException(status_code=403, detail="invalid_debug_key")


@router.get("/{run_id}/{filename}")
async def get_debug_file(run_id: str, filename: str, key: str = Query(...)):
    """读取 debug_output 下保存的可视化图片 / metrics.json。"""
    _verify_key(key)
    if ".." in run_id or ".." in filename:
        raise HTTPException(status_code=400, detail="invalid_path")

    import yaml

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_root = ROOT / cfg.get("debug", {}).get("output_dir", "debug_output")
    file_path = out_root / run_id / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="file_not_found")
    return FileResponse(file_path)
