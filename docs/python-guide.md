# Python 零基础指南 — iris-vision

本文说明 iris-vision 模块中**哪些文件可以改、哪些一般不用动**。

## 推荐环境

- Python **3.11**（MediaPipe 对 3.13 支持不稳定）
- 虚拟环境：项目目录下 `.venv/`

## 文件修改指南


| 文件/参数                             | 能否修改      | 说明                                    |
| --------------------------------- | --------- | ------------------------------------- |
| `config/grade_thresholds.yaml`    | **主要改这里** | L* 分档、基础颜色分类、`detection.mode`、瞳孔/环带参数 |
| `color_classification.`*（yaml 内）  | 可改        | CIELAB 基础颜色分类阈值，当前覆盖浅蓝/深蓝/绿/棕/深棕      |
| `eye_closeup.*`（yaml 内）           | 可改        | 瞳孔暗区百分位、环带相对瞳孔倍数                      |
| `app/services/grade.py`           | 可改        | 分档逻辑、置信度公式                            |
| `app/services/color.py`           | 可改        | Lab 取色、高光剔除、基础颜色分类逻辑                  |
| `app/services/quality.py` 内阈值     | 可改        | 模糊/过曝灵敏度                              |
| `app/services/eye_iris_detect.py` | 谨慎改       | 眼部特写瞳孔定位算法                            |
| `app/services/iris_detect.py`     | 谨慎改       | 定位模式调度（eye_closeup / face / auto）     |
| `main.py`                         | 一般不改      | 仅改端口时可动 uvicorn 参数                    |
| `app/models/schemas.py`           | 一般不改      | 响应字段结构，前后端对齐后再改                       |
| `app/api/routes.py`               | 一般不改      | HTTP 路由定义                             |
| `requirements.txt`                | 少改        | 仅增删依赖时                                |


## 处理流程（只读了解）

```
上传「眼部特写」→ 质量检测(模糊/过曝)
              → 瞳孔定位 → 虹膜环带取色
              → 去高光 → RGB 转 CIELAB → L*/a*/b* 中位数
              → 映射 Grade + 判断基础虹膜颜色
```

默认 **不需要全脸**：`detection.mode: eye_closeup`（见 `config/grade_thresholds.yaml`）。
若需兼容全脸照，改为 `face` 或 `auto`。

## Grade 与颜色识别

- `Grade 1–5` 仍只使用 `L`* 明度，目标是近似 Pan 2017 风格深浅分档。
- `iris_color` 使用 `L*/a*/b*` 做基础颜色判断，当前只覆盖论文色类中的纯色子集：浅蓝色、深蓝色、绿色、棕色、深棕色。
- 复合类型（例如蓝色带棕色瞳孔周环、绿色带棕色环、外绿内棕等）暂未加入，因为它们需要把完整虹膜分区后分别取色。
- 调试页与 Vue 页面会显示 `虹膜颜色：xx`，并在 debug 指标中展示颜色置信度和判定依据。

## 常用命令

```powershell
cd iris-vision
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

浏览器打开 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) 可在线测试 API。

## 标定步骤（后续）

1. 收集样张，人工按 Pan Figure 1 分 Grade 1–5
2. 记录每张的 L* 中位数
3. 调整 `grade_thresholds.yaml` 中的 `boundaries`
4. 对比自动分级与人工分级的一致性

