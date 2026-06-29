# iris-color

虹膜颜色识别系统 MVP — 仅识别虹膜颜色深浅（Grade 1–5），不做身份识别。

## 架构

```
iris-web (Vue 3)  →  iris-api (Spring Boot)  →  iris-vision (Python FastAPI)
   :5173                    :8080                        :8000
```

## 分级依据

参考 Pan et al. 2017（Ophthalmic Physiol Opt 2018;38:48–55，DOI: [10.1111/opo.12427](https://doi.org/10.1111/opo.12427)）Figure 1 的 5 档主观分级：

- Grade 1 = 最浅，Grade 5 = 最深
- 介于两档 → 取较高档（更深）
- MVP 使用 CIELAB L* 占位阈值自动近似，**需后续标定**

详见 [docs/pan2017-reference.md](docs/pan2017-reference.md)。

## 服务管理（启动 / 关闭 / 重启）

默认端口：`iris-vision` **8000** · `iris-api` **8080** · `iris-web` **5173**

> 开发时每个服务占一个终端窗口。**关闭**：在该窗口按 `Ctrl + C`。  
> 若窗口已关或进程残留，可用下方「按端口关闭」命令。

### 首次启动（按顺序）

#### 1. iris-vision（Python）

```powershell
cd iris-vision
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_model.py   # 仅首次需要
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

验证：http://127.0.0.1:8000/docs

#### 2. iris-api（Spring Boot）

```powershell
cd iris-api
.\mvnw spring-boot:run
```

验证：http://localhost:8080/api/v1/health

#### 3. iris-web（Vue 3）

```powershell
cd iris-web
npm install   # 仅首次需要
npm run dev
```

验证：http://localhost:5173 — 拍照或上传图片，查看 Grade 结果

---

### 日常启动（依赖已装好）

```powershell
# 终端 1
cd iris-vision
.\.venv\Scripts\activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 终端 2
cd iris-api
.\mvnw spring-boot:run

# 终端 3
cd iris-web
npm run dev
```

---

### 关闭

| 方式 | 说明 |
|------|------|
| **推荐** | 在对应终端按 `Ctrl + C` |
| **按端口** | 见下方 PowerShell 命令 |

关闭单个服务（将 `8000` 换成 `8080` / `5173`）：

```powershell
$p = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
if ($p) { $p | ForEach-Object { Stop-Process -Id $_ -Force } }
```

一键关闭全部三个服务：

```powershell
8000, 8080, 5173 | ForEach-Object {
  Get-NetTCPConnection -LocalPort $_ -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique |
    ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
}
```

---

### 重启

**原则**：先关后开；顺序仍为 vision → api → web（api 依赖 vision，web 依赖 api）。

```powershell
# 1. 关闭全部
8000, 8080, 5173 | ForEach-Object {
  Get-NetTCPConnection -LocalPort $_ -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique |
    ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
}

# 2. 等待端口释放
Start-Sleep -Seconds 2

# 3. 分别在三个新终端中按「日常启动」命令重新启动
```

只重启某一个服务时：先关闭对应端口，再执行该服务的启动命令。  
若只改了 Python 代码，`uvicorn --reload` 会自动热重载，一般无需手动重启 vision。

---

## 快速启动（首次安装摘要）

见上方「首次启动」三节；各模块详情见子目录 README。

## 目录结构

```
iris-color/
├── docs/           # 论文参考、Python 入门指南
├── iris-vision/    # 图像分析核心（FastAPI）
├── iris-api/       # 业务 API、转发（Spring Boot）
├── iris-web/       # 前端拍照页（Vue 3）
└── iris-desktop/   # Windows 桌面版（Electron 壳）
```

## Windows 桌面版

将系统打包为独立 Windows 软件（内置 JRE + Python，用户无需预装运行环境）。

### 构建安装包（开发者）

```powershell
cd iris-desktop
.\scripts\build-all.ps1
```

首次构建会自动下载 JRE 17 与 Embeddable Python 3.11 并安装 `iris-vision` 依赖（约 500–700 MB 安装包）。  
产出：`iris-desktop/dist/IrisColor Setup 0.1.0.exe`

### 分步构建

```powershell
# 1. 准备运行时（仅首次，或 requirements.txt / Java 版本变更后）
cd iris-desktop
.\scripts\prepare-jre.ps1
.\scripts\prepare-python.ps1

# 2. 构建含前端的 JAR
cd ..\iris-api
.\mvnw.cmd clean package -Pdesktop -DskipTests
copy target\iris-api-*.jar ..\iris-desktop\resources\iris-api.jar

# 3. 打包
cd ..\iris-desktop
npm install
npm run dist
```

> **PowerShell 中文乱码**：脚本已内置 UTF-8 设置（`scripts/_init.ps1`）。若仍乱码，可先执行 `chcp 65001`，或使用 Windows Terminal。
>
> **打包报「无法创建符号链接 / 客户端没有所需的特权」**：已在 `package.json` 中关闭代码签名（`signAndEditExecutable: false`）。若仍失败，可开启 Windows **开发者模式**（设置 → 隐私和安全性 → 开发者选项），或以管理员身份运行 PowerShell。

### 本地调试 Electron（不打包）

需已启动 vision / 已构建 JAR，或使用 `resources/` 中的运行时：

```powershell
cd iris-desktop
npm install
npm start
```

### 与网页部署的关系

- **日常开发**：仍用上方三终端 + 浏览器 `http://localhost:5173`，无需 Electron
- **网页部署**：仍按 [docs/deploy-server1.md](docs/deploy-server1.md)，`-Pdesktop` 仅桌面打包时启用
- 业务代码（web / api / vision）桌面版与网页版**共用同一份源码**

---

请阅读 [docs/python-guide.md](docs/python-guide.md)，了解哪些配置文件可以修改。

## 合规提醒

眼部照片属于敏感个人信息。MVP 阶段不落库；正式上线需知情同意、加密存储与权限控制。
