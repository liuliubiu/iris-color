#!/bin/bash
# 在每台服务器上执行一次，用于对比空闲度、已跑项目、占用端口。
# 用法: bash server-audit.sh
# 若从 Windows 上传后出现 $'\r': command not found，先执行:
#   sed -i 's/\r$//' server-audit.sh

HOST=$(hostname 2>/dev/null || echo unknown)
NOW=$(date '+%Y-%m-%d %H:%M:%S')

echo "========== SERVER AUDIT =========="
echo "主机名: $HOST"
echo "时间:   $NOW"
echo ""

echo "---------- 1. 负载与资源（越低越空闲）----------"
if command -v uptime >/dev/null 2>&1; then
  uptime
fi
echo ""
if command -v free >/dev/null 2>&1; then
  free -h
fi
echo ""
if command -v df >/dev/null 2>&1; then
  df -h / /opt /home 2>/dev/null | sort -u || df -h /
fi
echo ""
if [ -r /proc/loadavg ]; then
  echo "loadavg(1/5/15): $(cat /proc/loadavg)"
fi
if command -v nproc >/dev/null 2>&1; then
  echo "CPU 逻辑核数: $(nproc)"
fi
echo ""

echo "---------- 2. 监听端口（LISTEN）----------"
if command -v ss >/dev/null 2>&1; then
  ss -tlnp 2>/dev/null || ss -tln
elif command -v netstat >/dev/null 2>&1; then
  netstat -tlnp 2>/dev/null || netstat -tln
else
  echo "未找到 ss/netstat"
fi
echo ""

echo "---------- 3. iris-color 相关端口是否占用 ----------"
_listen_check() {
  local p=$1
  if command -v ss >/dev/null 2>&1; then
    ss -tln 2>/dev/null | grep -q ":${p} "
    return $?
  fi
  if command -v netstat >/dev/null 2>&1; then
    netstat -tln 2>/dev/null | grep -q ":${p} "
    return $?
  fi
  return 1
}
for p in 80 443 8080 8000 5173 3306; do
  if _listen_check "$p"; then
    echo "  端口 $p : 已占用"
  else
    echo "  端口 $p : 空闲"
  fi
done
echo ""

echo "---------- 4. 常见项目进程（Java / Node / Python / Nginx）----------"
ps aux 2>/dev/null | grep -E '[j]ava|[n]ode|[p]ython|[u]vicorn|[g]unicorn|[n]ginx' | grep -v grep || echo "（无匹配或无权查看）"
echo ""

echo "---------- 5. systemd 已启用业务服务（非系统服务摘要）----------"
if command -v systemctl >/dev/null 2>&1; then
  systemctl list-units --type=service --state=running 2>/dev/null \
    | grep -vE 'systemd-|ssh|cron|rsyslog|network|dbus|getty|user@|polkit|tuned|chrony|firewalld|auditd|irqbalance|kdump|microcode|qemu|cloud-' \
    | head -40 || true
else
  echo "无 systemctl"
fi
echo ""

echo "---------- 6. /opt /home 下常见部署目录 ----------"
for d in /opt /home/*/app /home/*/projects /www /var/www; do
  if [ -d "$d" ] 2>/dev/null; then
    ls -la "$d" 2>/dev/null | head -20
  fi
done
echo ""

echo "---------- 7. 正在运行的 jar 路径（Spring Boot）----------"
ps aux 2>/dev/null | grep -E '[j]ar' | grep -v grep || echo "（未发现 java -jar）"
echo ""

echo "========== END $HOST =========="
