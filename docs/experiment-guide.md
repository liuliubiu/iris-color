# 实验记录子系统（本地开发专用）

在 `iris-vision` 内嵌的实验数据管理工具。支持 **MySQL**（推荐本地使用）或 **SQLite**，默认关闭，不影响云端 `/analyze` 生产链路。

## 启用

编辑 [`iris-vision/config/grade_thresholds.yaml`](../iris-vision/config/grade_thresholds.yaml)：

```yaml
experiments:
  enabled: true          # 本地 true；云端保持 false
  api_key: iris-color-dev
  backend: mysql         # mysql | sqlite
  mysql:
    host: 127.0.0.1
    port: 3306
    user: root
    password: ""         # 按你的 MySQL 密码填写
    database: iris_experiment
    charset: utf8mb4
```

## MySQL 初始化

1. 确保本地 MySQL 已启动
2. 执行初始化脚本（或让服务首次连接时自动建表）：

```bash
mysql -u root -p < iris-vision/scripts/init_experiment_mysql.sql
```

3. 安装依赖：`pip install pymysql`

## 访问

```
http://127.0.0.1:8000/experiments/ui?key=iris-color-dev
```

前端依赖（Vue / Element Plus）已内置在 `app/static/vendor/`，无需外网 CDN。

## 与 Debug 调试台联动

1. 在 [Debug 调试台](http://127.0.0.1:8000/debug/ui?key=iris-color-dev) 从 `img/` 列表载入图片并识别（需 `debug.save_to_disk: true` 以保存 run）
2. 识别完成后点击 **「加入实验记录」**，或到实验记录页点击 **「从 Debug 导入」**
3. 系统自动填入 Grade、L*、颜色及图片/Debug 关联；您只需补充实验组、实验人、设备等

从 img 列表载入的图片会写入 `source_rel`，保存到实验记录的 `image_rel` 字段。

### 批量识别并导入（整文件夹）

适用于同一批次有多张图片、需一次性归入同一大组/小组的场景：

1. 在 Debug 调试台 **「实验记录 · 组别」** 区域填写大组、小组、实验人、日期等（与单张导入相同）
2. 在 **「批量识别并导入」** 中选择 `img/` 子文件夹（或 img 根目录）
3. 可选：**跳过已导入的图片**（默认开启）、**包含子文件夹**
4. 点击 **「批量识别并导入本文件夹」**，确认后系统自动识别并写入实验记录
5. 完成后弹窗展示成功 / 失败 / 跳过汇总，可跳转实验记录页查看

**质量检测**：未勾选「跳过质量检测」时，若某张图未通过模糊/过曝检测，系统会自动跳过质量门槛继续识别（与单条重识别行为一致），并在结果中标注。

**识别区域**：导入成功后，在实验记录页点击 **「识别区域 → 查看」** 可查看虹膜环带定位图；若区域有误，可点 **「人工调整重识别」** 或直接 **「删除此记录」**。

> 批量导入会直接创建记录，无需逐张打开实验记录编辑抽屉。单张「加入实验记录」流程仍可使用。

## 命名规范（序号式）

| 字段 | 格式 | 示例 | 说明 |
|------|------|------|------|
| 实验大组 | `G{YYYYMMDD}-{NNN}` | `G20260722-001` | 当日第 1 个大组实验 |
| 实验小组 | `S{NN}` | `S01` | 大组内子批次；不分组则留空 |

新建时可点「自动编号」获取下一个建议序号。

## 颜色（9 种固定值）

- 蓝色系：浅蓝、蓝、深蓝
- 绿色系：浅绿、绿、深绿
- 棕色系：浅棕、棕、深棕

## 界面操作

- **表头筛选栏**：横向排列于表格上方，支持多字段联合筛选
- **筛选 / 重置所有筛选**：应用或清空全部条件
- **新建记录**：自动填入建议的大组/小组编号；保存后自动刷新并清除筛选，确保新记录可见
- 双击行或点「编辑」修改；支持批量删除

## REST API

Prefix：`/experiments`，均需 `?key=`。

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/experiments/ui` | 管理页 |
| GET | `/experiments/records` | 列表 |
| POST | `/experiments/records` | 新建 |
| PUT | `/experiments/records/{id}` | 更新 |
| DELETE | `/experiments/records/{id}` | 删除 |
| POST | `/experiments/records/bulk-delete` | 批量删除 |
| POST | `/experiments/import/batch-from-folder` | 批量识别 img/ 文件夹并自动导入实验记录 |
| GET | `/experiments/meta/options` | 枚举与历史值 |
| GET | `/experiments/meta/suggest-names` | 建议大组/小组编号 |

## 云端部署

- `experiments.enabled: false`
- 不上传 MySQL 配置中的本地密码
- 实验 API 不经过 Nginx / iris-api
