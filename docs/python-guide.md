# Python 零基础指南 — iris-vision

本文说明 iris-vision 模块中**哪些文件可以改、哪些一般不用动**。

## 推荐环境

- Python **3.11**（MediaPipe 对 3.13 支持不稳定）
- 虚拟环境：项目目录下 `.venv/`

## 文件修改指南

| 文件/参数 | 能否修改 | 说明 |
|-----------|----------|------|
| `config/grade_thresholds.yaml` | **主要改这里** | L* 分档阈值，标定后更新 |
| `app/services/grade.py` | 可改 | 分档逻辑、置信度公式 |
| `app/services/quality.py` 内阈值 | 可改 | 模糊/过曝/闭眼灵敏度 |
| `app/services/iris_detect.py` 环带比例 | 谨慎改 | 虹膜采样环半径（默认内径 30%、外径 80%） |
| `main.py` | 一般不改 | 仅改端口时可动 uvicorn 参数 |
| `app/models/schemas.py` | 一般不改 | 响应字段结构，前后端对齐后再改 |
| `app/api/routes.py` | 一般不改 | HTTP 路由定义 |
| `requirements.txt` | 少改 | 仅增删依赖时 |

## 处理流程（只读了解）

```
上传图片 → 质量检测 → MediaPipe 定位眼部 → 虹膜环带取色
         → 去高光 → RGB 转 CIELAB → 计算 L*/a*/b* 中位数 → 映射 Grade
```

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
