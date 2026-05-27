# iris-api

Spring Boot 业务 API，接收前端上传并转发至 iris-vision。默认端口 **8080**。

需先启动 iris-vision（8000）。

## 启动

```powershell
cd iris-api
.\mvnw spring-boot:run
```

## 关闭

- 在运行 Maven 的终端按 **`Ctrl + C`**
- 或按端口强制结束：

```powershell
$p = Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
if ($p) { $p | ForEach-Object { Stop-Process -Id $_ -Force } }
```

## 重启

```powershell
# 关闭
$p = Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
if ($p) { $p | ForEach-Object { Stop-Process -Id $_ -Force } }
Start-Sleep -Seconds 2

# 启动（确保 iris-vision 已在 8000 运行）
cd iris-api
.\mvnw spring-boot:run
```

修改 `application.properties` 或 Java 代码后需重启本服务。

## 接口

- `GET /api/v1/health` — 健康检查（含 vision 状态）
- `POST /api/v1/iris/analyze` — 上传图片（multipart `file`），返回 Lab + Grade

## 配置

`src/main/resources/application.properties`：

- `server.port=8080`
- `iris.vision.base-url=http://127.0.0.1:8000`
