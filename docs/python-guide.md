# Python 零基础指南 — iris-vision

本文说明 iris-vision 模块中**哪些文件可以改、哪些一般不用动**。

## 推荐环境

- Python **3.11**（MediaPipe 对 3.13 支持不稳定）
- 虚拟环境：项目目录下 `.venv/`

## 文件修改指南

| 文件/参数 | 能否修改 | 说明 |
|-----------|----------|------|
| `config/grade_thresholds.yaml` | **主要改这里** | L* 分档、`detection.mode`、瞳孔/环带参数 |
| `eye_closeup.*`（yaml 内） | 可改 | 瞳孔暗区百分位、环带相对瞳孔倍数 |
| `app/services/grade.py` | 可改 | 分档逻辑、置信度公式 |
| `app/services/quality.py` 内阈值 | 可改 | 模糊/过曝灵敏度 |
| `app/services/eye_iris_detect.py` | 谨慎改 | 眼部特写瞳孔定位算法 |
| `app/services/iris_detect.py` | 谨慎改 | 定位模式调度（eye_closeup / face / auto） |
| `main.py` | 一般不改 | 仅改端口时可动 uvicorn 参数 |
| `app/models/schemas.py` | 一般不改 | 响应字段结构，前后端对齐后再改 |
| `app/api/routes.py` | 一般不改 | HTTP 路由定义 |
| `requirements.txt` | 少改 | 仅增删依赖时 |

## 处理流程（只读了解）

```
上传「眼部特写」→ 质量检测(模糊/过曝)
              → 瞳孔定位 → 虹膜环带取色
              → 去高光 → RGB 转 CIELAB → L*/a*/b* 中位数 → 映射 Grade
```

默认 **不需要全脸**：`detection.mode: eye_closeup`（见 `config/grade_thresholds.yaml`）。
若需兼容全脸照，改为 `face` 或 `auto`。

## 常用命令

```powershell
cd iris-vision
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

浏览器打开 http://127.0.0.1:8000/docs 可在线测试 API。

## 标定步骤（后续）

1. 收集样张，人工按 Pan Figure 1 分 Grade 1–5
2. 记录每张的 L* 中位数
3. 调整 `grade_thresholds.yaml` 中的 `boundaries`
4. 对比自动分级与人工分级的一致性
