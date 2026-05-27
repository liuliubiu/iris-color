# iris-vision

Python FastAPI 虹膜颜色分析服务。默认端口 **8000**。

## 启动

```powershell
cd iris-vision
.\.venv\Scripts\activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

首次安装：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_model.py
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 关闭

- 在运行 uvicorn 的终端按 **`Ctrl + C`**
- 或按端口强制结束：

```powershell
$p = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
if ($p) { $p | ForEach-Object { Stop-Process -Id $_ -Force } }
```

## 重启

```powershell
# 关闭
$p = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
if ($p) { $p | ForEach-Object { Stop-Process -Id $_ -Force } }
Start-Sleep -Seconds 1

# 启动
cd iris-vision
.\.venv\Scripts\activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

修改 `config/` 或 `app/services/` 后需重启；仅改代码且带 `--reload` 时通常会自动重载。

## API

- `GET /health` — 健康检查
- `POST /analyze` — 上传图片，返回 Lab + Grade

在线文档：http://127.0.0.1:8000/docs

## 配置

分档阈值见 `config/grade_thresholds.yaml`（标定时主要修改此文件）。

Python 新手请阅读上级目录 [docs/python-guide.md](../docs/python-guide.md)。
