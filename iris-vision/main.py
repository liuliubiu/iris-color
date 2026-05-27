"""
iris-vision 入口文件。

启动方式（在 iris-vision 目录下）：
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

一般不需要修改本文件，除非要改端口或 CORS 设置。
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.debug_files import router as debug_files_router
from app.api.debug_routes import router as debug_router
from app.api.routes import router

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
