# 调试后门（开发/标定专用）

**推荐入口**：浏览器直接打开调试台（拍照/上传 + 各阶段图 + 参数，无需手动调 API）

```
http://127.0.0.1:8000/debug/ui?key=iris-color-dev
```

或简写：`http://127.0.0.1:8000/debug`（自动跳转到 `/debug/ui`）

## 调试台功能

- 摄像头拍照 / 本地文件上传
- 可选「跳过质量检测」
- 一次识别后页面内展示：
  - Grade、L\*/a\*/b\*、瞳孔/环带半径、像素数等
  - 原图、瞳孔定位、虹膜环带、去高光、有效取色、mask 共 6 张图
- 同时保存到 `iris-vision/debug_output/{run_id}/`

## 密钥

默认 `iris-color-dev`，可在页面顶部修改，或在 `config/grade_thresholds.yaml`：

```yaml
debug:
  api_key: iris-color-dev
```

## 备用：纯 API

`POST /debug/analyze`，Header `X-Debug-Key` 或 query `?key=...`

Swagger：http://127.0.0.1:8000/docs → **debug** 分组

## 各阶段图片说明

| 图 | 含义 |
|----|------|
| `01_pupil_localization` | 黄框=搜索区；蓝圆=瞳孔；红十字=中心 |
| `02_iris_ring` | 绿=外缘；红=内缘；青=采样环 |
| `03_highlight_rejection` | 红=环内被剔除的高光 |
| `04_valid_samples` | 绿=参与 Lab 中位数的像素 |
| `05_ring_mask_only` | 环带 mask 伪彩色 |

## PowerShell（可选）

```powershell
$k = "iris-color-dev"
Invoke-RestMethod -Uri "http://127.0.0.1:8000/debug/analyze?key=$k&skip_quality=true" `
  -Method Post -Headers @{ "X-Debug-Key" = $k } `
  -Form @{ file = Get-Item "D:\path\to\eye.jpg" }
```
