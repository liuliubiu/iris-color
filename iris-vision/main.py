"""
iris-vision 入口文件。

启动方式（在 iris-vision 目录下）：
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

一般不需要修改本文件，除非要改端口或 CORS 设置。
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from app.api.debug_files import router as debug_files_router
from app.api.debug_routes import router as debug_router
from app.api.label_routes import router as label_router
from app.api.routes import router
from app.services.pipeline import load_config

app = FastAPI(
    title="iris-vision",
    description="虹膜颜色识别服务 — CIELAB 取色 + Pan 2017 风格 Grade 分档",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(debug_router)
app.include_router(debug_files_router)
app.include_router(label_router)

_config_path = Path(__file__).parent / "config" / "grade_thresholds.yaml"
if load_config(_config_path).get("experiments", {}).get("enabled", False):
    from fastapi.staticfiles import StaticFiles

    from app.api.experiment_routes import router as experiment_router

    _vendor_dir = Path(__file__).parent / "app" / "static" / "vendor"
    if _vendor_dir.is_dir():
        app.mount(
            "/experiments/vendor",
            StaticFiles(directory=_vendor_dir),
            name="experiment_vendor",
        )
    app.include_router(experiment_router)
