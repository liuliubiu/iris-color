# iris-desktop

IrisColor Windows 桌面版 — Electron 壳层，自动启动 `iris-vision` + `iris-api` 并打开原生窗口。

## 快速构建

```powershell
.\scripts\build-all.ps1
```

## 本地调试

```powershell
npm install
npm start
```

开发模式下会依次尝试：

- `resources/python/python.exe` → `iris-vision/.venv` → 系统 `python`
- `resources/jre/bin/java.exe` → 系统 `java`
- `resources/iris-api.jar` → `../iris-api/target/iris-api-0.0.1-SNAPSHOT.jar`

## 目录

| 路径 | 说明 |
|------|------|
| `electron/main.js` | 主进程：子进程管理、健康检查、窗口 |
| `scripts/prepare-jre.ps1` | 下载 Temurin JRE 17 |
| `scripts/prepare-python.ps1` | 下载 Embeddable Python + 依赖 |
| `scripts/build-all.ps1` | 一键构建安装包 |
| `resources/` | 构建时填充（JRE / Python / JAR），不提交 git |
