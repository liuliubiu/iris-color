# iris-web

Vue 3 前端 — 拍照/上传并展示虹膜颜色 Grade 结果。默认端口 **5173**。

需先启动 iris-vision（8000）和 iris-api（8080）。

## 启动

```powershell
cd iris-web
npm run dev
```

首次安装：`npm install`

浏览器打开 http://localhost:5173

## 关闭

- 在运行 Vite 的终端按 **`Ctrl + C`**
- 或按端口强制结束：

```powershell
$p = Get-NetTCPConnection -LocalPort 5173 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
if ($p) { $p | ForEach-Object { Stop-Process -Id $_ -Force } }
```

## 重启

```powershell
# 关闭
$p = Get-NetTCPConnection -LocalPort 5173 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
if ($p) { $p | ForEach-Object { Stop-Process -Id $_ -Force } }
Start-Sleep -Seconds 1

# 启动
cd iris-web
npm run dev
```

修改 `vite.config.ts` 或环境变量后需重启；改 `.vue` 文件时 Vite 会自动热更新。

## 开发代理

Vite 将 `/api` 代理至 `http://localhost:8080`。
