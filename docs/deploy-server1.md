# iris-color 部署指南（服务器 1）

- **主机**：119.91.207.69  
- **目录**：`/home/server/iris-color`  
- **端口**：iris-api **8090**，iris-vision **8003**（仅本机监听，不对外暴露）  
- **对外**：沿用现有 Nginx 的 **80 / 443** + **域名**

---

## 一、整体结构

```
浏览器 → 域名(DNS) → Nginx:443/80 → dist 静态页
                              └→ /api/* → 127.0.0.1:8090 (iris-api)
                                              └→ 127.0.0.1:8003 (iris-vision)
```

前端只请求 `/api/v1/...`，**不用改前端代码**；Nginx 把 `/api` 转到 8090 即可。

---

## 二、域名要怎么做？（不能“只填域名就用”）

需要 **三步都做完** 才能用 `https://iris.公司域名.com` 访问：

| 步骤 | 谁来做 | 做什么 |
|------|--------|--------|
| 1. DNS | 域名管理员（公司 IT 或域名控制台） | 新增 **A 记录**：例如 `iris` → `119.91.207.69`（或完整子域名 `iris.xxx.com` 指向该 IP） |
| 2. Nginx | 你在服务器 1 上 | 增加一个 `server { server_name iris.xxx.com; ... }`，指到 `/home/server/iris-color/iris-web/dist`，`/api` 反代到 `8090` |
| 3. HTTPS 证书 | 服务器上（与公司其它站相同方式） | 若其它项目已是 HTTPS，通常用 **同一套 certbot/公司证书流程** 给新子域名签证书 |

**不能**只在项目里写域名而不配 DNS——浏览器解析不到 IP。  
**不能**只配 DNS 而不配 Nginx——请求会落到默认站点或 404。  

建议：向同事要一个 **未占用的子域名**（如 `iris.现有主域名.com`），参考同机上已有项目的 `server_name` 和 `ssl_certificate` 写法，复制改路径即可。

---

## 三、服务器目录（首次）

```bash
mkdir -p /home/server/iris-color/{iris-vision,iris-api,iris-web/dist}
```

最终结构：

```
/home/server/iris-color/
├── start.sh / stop.sh    # 方式 A 启停脚本（五.2 创建）
├── logs/                 # 方式 A 运行日志与 pid（五.2 创建）
├── iris-vision/          # Python 整目录（含 config、assets、.venv）
├── iris-api/
│   └── iris-api-0.0.1-SNAPSHOT.jar
└── iris-web/
    └── dist/             # npm run build 产物
```

---

## 四、本机（Windows）构建与上传

### 1. iris-web

```powershell
cd d:\project\iris-color\iris-web
npm install
npm run build
```

XFTP：把 `iris-web\dist\` **里所有文件** 上传到 `/home/server/iris-color/iris-web/dist/`。

### 2. iris-api

```powershell
cd d:\project\iris-color\iris-api
.\mvnw.cmd clean package -DskipTests
```

XFTP：上传 `iris-api\target\iris-api-0.0.1-SNAPSHOT.jar` → `/home/server/iris-color/iris-api/`。

启动时使用 **`prod` 配置**（端口 8090、vision 8003），见下文 **五.2**。

### 3. iris-vision

XFTP：上传整个 `iris-vision` 目录（排除 `.venv`、`__pycache__` 可减小体积；或在服务器上重新建 venv）。

**模型文件**：本地若已有 `iris-vision/assets/models/face_landmarker.task`，一并上传；否则在服务器执行 `python scripts/download_model.py`（需能访问外网）。

---

## 五、服务器首次安装（XShell，只做一次）

### 1. Python 环境

```bash
cd /home/server/iris-color/iris-vision
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python scripts/download_model.py   # 若未上传模型
deactivate
```

确认 venv 已建好（隐藏目录，普通 `ls` 看不到）：

```bash
ls -la /home/server/iris-color/iris-vision/.venv/bin/python
```

### 2. 启动 iris-vision + iris-api（二选一）

`application-prod.properties` 已写在仓库里（8090 + 8003），无需再手写端口。  
**顺序固定**：先 **iris-vision（8003）**，再 **iris-api（8090）**。

| | 方式 A：手动 / 脚本 | 方式 B：systemd |
|--|---------------------|-----------------|
| 适用 | 首次试跑、不想动系统服务 | 长期生产、开机自启 |
| 关机/SSH 断开 | 用 `nohup` 或下方脚本可保持运行 | 由 systemd 托管 |
| 崩溃后 | 需手动重启 | 可自动 `Restart` |

---

#### 方式 A：不用 systemd（推荐先试跑）

##### A.1 首次：创建启停脚本（在 `iris-color` 根目录）

```bash
mkdir -p /home/server/iris-color/logs

cat > /home/server/iris-color/start.sh << 'EOF'
#!/bin/bash
set -e
ROOT="/home/server/iris-color"
cd "$ROOT"

mkdir -p logs

# iris-vision (8003)
if [ -f logs/vision.pid ] && kill -0 "$(cat logs/vision.pid)" 2>/dev/null; then
  echo "iris-vision 已在运行 (pid $(cat logs/vision.pid))"
else
  cd iris-vision
  nohup .venv/bin/uvicorn main:app --host 127.0.0.1 --port 8003 \
    > ../logs/vision.log 2>&1 &
  echo $! > ../logs/vision.pid
  cd ..
  echo "iris-vision 已启动 (pid $(cat logs/vision.pid))"
fi

sleep 2

# iris-api (8090)
if [ -f logs/api.pid ] && kill -0 "$(cat logs/api.pid)" 2>/dev/null; then
  echo "iris-api 已在运行 (pid $(cat logs/api.pid))"
else
  cd iris-api
  nohup java -Xms128m -Xmx256m -jar iris-api-0.0.1-SNAPSHOT.jar \
    --spring.profiles.active=prod > ../logs/api.log 2>&1 &
  echo $! > ../logs/api.pid
  cd ..
  echo "iris-api 已启动 (pid $(cat logs/api.pid))"
fi

sleep 2
curl -sf http://127.0.0.1:8003/health && echo "  vision OK" || echo "  vision FAIL — 见 logs/vision.log"
curl -sf http://127.0.0.1:8090/api/v1/health && echo "  api OK" || echo "  api FAIL — 见 logs/api.log"
EOF

cat > /home/server/iris-color/stop.sh << 'EOF'
#!/bin/bash
ROOT="/home/server/iris-color"
cd "$ROOT"

stop_pid() {
  local name=$1 file=$2
  if [ -f "$file" ]; then
    local pid
    pid=$(cat "$file")
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      echo "$name 已停止 (pid $pid)"
    else
      echo "$name 未在运行 (pid 文件过期)"
    fi
    rm -f "$file"
  else
    echo "$name 无 pid 文件，跳过"
  fi
}

# 先停 api，再停 vision
stop_pid iris-api logs/api.pid
sleep 1
stop_pid iris-vision logs/vision.pid
EOF

chmod +x /home/server/iris-color/start.sh /home/server/iris-color/stop.sh
sed -i 's/\r$//' /home/server/iris-color/start.sh /home/server/iris-color/stop.sh
```

##### A.2 首次启动与验证

```bash
cd /home/server/iris-color
./start.sh
```

日志位置：

- `logs/vision.log` — Python 服务
- `logs/api.log` — Java 服务

关闭：

```bash
cd /home/server/iris-color
./stop.sh
```

重启（例如更新了 jar 或 Python 代码）：

```bash
cd /home/server/iris-color
./stop.sh
./start.sh
```

##### A.3 前台调试（两个 XShell 窗口，关窗口即停）

适合排查启动报错；**不要**与 `./start.sh` 同时跑，否则会端口冲突。

**窗口 1 — iris-vision：**

```bash
cd /home/server/iris-color/iris-vision
source .venv/bin/activate
uvicorn main:app --host 127.0.0.1 --port 8003
```

**窗口 2 — iris-api：**

```bash
cd /home/server/iris-color/iris-api
java -Xms128m -Xmx256m -jar iris-api-0.0.1-SNAPSHOT.jar --spring.profiles.active=prod
```

验证：`curl http://127.0.0.1:8003/health` 与 `curl http://127.0.0.1:8090/api/v1/health`  
停止：在对应窗口按 `Ctrl + C`（先 api，后 vision）。

##### A.4 手动方式注意

- `.venv` 是隐藏目录，用 `ls -la iris-vision` 查看，不是 `venv`。
- 服务器**重启后**不会自动拉起，需再执行 `./start.sh`。
- 若 `./start.sh` 报端口占用：`ss -tlnp | grep -E '8003|8090'`，确认旧进程后 `./stop.sh` 或 `kill` 对应 pid。
- 若从 Windows 粘贴命令出现 `$'\E[200~'` 乱码，请**手打** `cd` 路径，或分段粘贴。

---

#### 方式 B：systemd（长期生产）

##### B.1 iris-vision（8003）

```bash
cat > /etc/systemd/system/iris-vision.service << 'EOF'
[Unit]
Description=iris-vision
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/home/server/iris-color/iris-vision
ExecStart=/home/server/iris-color/iris-vision/.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8003
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable iris-vision
systemctl start iris-vision
curl -s http://127.0.0.1:8003/health
```

##### B.2 iris-api（8090）

```bash
cat > /etc/systemd/system/iris-api.service << 'EOF'
[Unit]
Description=iris-api
After=iris-vision.service
Requires=iris-vision.service

[Service]
Type=simple
User=root
WorkingDirectory=/home/server/iris-color/iris-api
ExecStart=/usr/bin/java -Xms128m -Xmx256m -jar /home/server/iris-color/iris-api/iris-api-0.0.1-SNAPSHOT.jar --spring.profiles.active=prod
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable iris-api
systemctl start iris-api
curl -s http://127.0.0.1:8090/api/v1/health
```

日常：`systemctl start|stop|restart iris-vision` / `iris-api`。  
若已用 **方式 A** 的 `./start.sh` 跑起来，**不要**再 `systemctl start`，否则会双开抢端口；先 `./stop.sh` 再切到 systemd。

---

### 3. Nginx（域名 + HTTPS）

先查看现有配置位置：

```bash
/usr/local/nginx/sbin/nginx -t
# 常见：/usr/local/nginx/conf/nginx.conf 或 conf/vhost/*.conf
```

在**不影响现有站点**的前提下，新增站点（把 `iris.你的公司域名.com` 换成实际子域名）：

```nginx
server {
    listen 80;
    server_name iris.你的公司域名.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name iris.你的公司域名.com;

    # 证书路径照抄同机其它 HTTPS 站点，或 certbot 生成后填入
    ssl_certificate     /path/to/fullchain.pem;
    ssl_certificate_key /path/to/privkey.pem;

    root /home/server/iris-color/iris-web/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8090;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        client_max_body_size 10m;
        proxy_read_timeout 120s;
    }
}
```

检查并重载：

```bash
/usr/local/nginx/sbin/nginx -t && /usr/local/nginx/sbin/nginx -s reload
```

**证书**：若公司用 certbot，可向运维确认后执行（示例）：

```bash
certbot certonly --nginx -d iris.你的公司域名.com
```

具体以公司现有流程为准。

---

## 六、日常更新（和你们其它项目一样）

| 改了什么 | 本机 | XFTP | 服务器命令（手动） | 服务器命令（systemd） |
|----------|------|------|-------------------|----------------------|
| 前端 | `npm run build` | 覆盖 `iris-web/dist/` | 一般不用重启；Nginx reload 可选 | 同左 |
| Java | `mvnw package` | 覆盖 jar | `./stop.sh && ./start.sh` 或只杀 api 再启 | `systemctl restart iris-api` |
| Python / yaml | — | 上传对应文件 | `./stop.sh && ./start.sh` 或只重启 vision | `systemctl restart iris-vision` |

顺序：**先 vision，再 api**。

**手动（方式 A）：**

```bash
cd /home/server/iris-color
./stop.sh
./start.sh
```

**systemd（方式 B）：**

```bash
systemctl restart iris-vision
sleep 2
systemctl restart iris-api
```

---

## 七、验证清单

在服务器上：

```bash
curl http://127.0.0.1:8003/health
curl http://127.0.0.1:8090/api/v1/health
ss -tln | grep -E '8003|8090'    # 应只有 127.0.0.1 或 * 监听，勿与 8001/8002/8097 冲突
```

浏览器打开：`https://iris.你的公司域名.com`，拍照或上传，看是否返回 Grade。

---

## 八、常见问题

**Q：没有子域名，只能用 IP？**  
可临时用 IP 访问：Nginx 里 `server_name` 填 IP 或 `_`，用 `http://119.91.207.69`（需与安全组放行 80 一致）。正式环境仍建议子域名 + HTTPS。

**Q：8090/8003 要改防火墙吗？**  
若只监听 `127.0.0.1`，**不需要**对公网开放 8090/8003；只暴露 80/443。

**Q：脚本上传后 bash 报错 `$'\r'`？**  
执行：`sed -i 's/\r$//' 脚本名.sh`

**Q：内存不够？**  
本机已有 3 个 Java；若 vision 启动失败，用 `free -h` 查看，必要时向运维申请升配或减小其它 jar 内存。

**Q：不用 systemd，和用 systemd 有什么区别？**  
功能相同；差别是手动方式不会开机自启、进程崩溃需自己 `./start.sh`。试跑、临时部署用手动即可；稳定上线后可改用 **五.2 方式 B**。

**Q：`start.sh` 显示 FAIL？**  
`tail -50 /home/server/iris-color/logs/vision.log` 或 `logs/api.log` 看报错；常见原因：未在 `iris-vision` 下建 `.venv`、jar 未上传、8003/8090 被占用。

---

## 九、端口对照（服务器 1 已占用，勿改）

| 端口 | 现状 |
|------|------|
| 8001、8002、8097 | 其它 Java 项目 |
| 80、443 | Nginx（iris 共用，用不同 `server_name` 区分） |
| **8003、8090** | **iris-color（本部署）** |
