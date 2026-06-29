const path = require('path')
const fs = require('fs')
const http = require('http')
const { app, BrowserWindow, session, dialog } = require('electron')
const { spawn } = require('child_process')

const VISION_PORT = 8000
const API_PORT = 8080
const APP_URL = `http://127.0.0.1:${API_PORT}`
const VISION_HEALTH = `http://127.0.0.1:${VISION_PORT}/health`
const API_HEALTH = `${APP_URL}/api/v1/health`

/** @type {import('child_process').ChildProcess | null} */
let visionProcess = null
/** @type {import('child_process').ChildProcess | null} */
let apiProcess = null
/** @type {BrowserWindow | null} */
let mainWindow = null
/** @type {BrowserWindow | null} */
let splashWindow = null

function isPackaged() {
  return app.isPackaged
}

function getResourcesRoot() {
  if (isPackaged()) {
    return process.resourcesPath
  }
  return path.join(__dirname, '..', 'resources')
}

function getVisionDir() {
  if (isPackaged()) {
    return path.join(process.resourcesPath, 'iris-vision')
  }
  return path.join(__dirname, '..', '..', 'iris-vision')
}

function getPythonExe() {
  const bundled = path.join(getResourcesRoot(), 'python', 'python.exe')
  if (fs.existsSync(bundled)) {
    return bundled
  }
  const devVenv = path.join(getVisionDir(), '.venv', 'Scripts', 'python.exe')
  if (fs.existsSync(devVenv)) {
    return devVenv
  }
  return 'python'
}

function getJavaExe() {
  const bundled = path.join(getResourcesRoot(), 'jre', 'bin', 'java.exe')
  if (fs.existsSync(bundled)) {
    return bundled
  }
  return 'java'
}

function getApiJar() {
  const bundled = path.join(getResourcesRoot(), 'iris-api.jar')
  if (fs.existsSync(bundled)) {
    return bundled
  }
  const devJar = path.join(__dirname, '..', '..', 'iris-api', 'target', 'iris-api-0.0.1-SNAPSHOT.jar')
  if (fs.existsSync(devJar)) {
    return devJar
  }
  return bundled
}

function killProcessTree(proc) {
  if (!proc || proc.killed) return
  if (process.platform === 'win32') {
    spawn('taskkill', ['/pid', String(proc.pid), '/f', '/t'], { stdio: 'ignore' })
  } else {
    proc.kill('SIGTERM')
  }
}

function httpGet(url) {
  return new Promise((resolve, reject) => {
    const req = http.get(url, (res) => {
      res.resume()
      if (res.statusCode && res.statusCode >= 200 && res.statusCode < 300) {
        resolve(true)
      } else {
        reject(new Error(`HTTP ${res.statusCode}`))
      }
    })
    req.on('error', reject)
    req.setTimeout(3000, () => {
      req.destroy(new Error('timeout'))
    })
  })
}

async function waitForHealth(url, label, timeoutMs = 120000, intervalMs = 500) {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    try {
      await httpGet(url)
      return
    } catch {
      await new Promise((r) => setTimeout(r, intervalMs))
    }
  }
  throw new Error(`${label} 启动超时: ${url}`)
}

async function isPortInUse(port) {
  return new Promise((resolve) => {
    const server = http.createServer()
    server.once('error', () => resolve(true))
    server.once('listening', () => {
      server.close(() => resolve(false))
    })
    server.listen(port, '127.0.0.1')
  })
}

function createSplashWindow() {
  splashWindow = new BrowserWindow({
    width: 480,
    height: 280,
    frame: false,
    resizable: false,
    center: true,
    alwaysOnTop: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  })

  const html = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <style>
    body {
      margin: 0;
      font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
      background: linear-gradient(135deg, #1a4a5e, #2d8fad);
      color: #fff;
      display: flex;
      align-items: center;
      justify-content: center;
      height: 100vh;
    }
    .box { text-align: center; padding: 24px; }
    h1 { font-size: 22px; margin: 0 0 12px; font-weight: 600; }
    p { margin: 0; opacity: 0.9; font-size: 14px; }
    .dot { animation: pulse 1.2s infinite; }
    @keyframes pulse { 0%,100%{opacity:.4} 50%{opacity:1} }
  </style>
</head>
<body>
  <div class="box">
    <h1>豪赋-虹膜颜色识别</h1>
    <p class="dot">正在启动服务，请稍候…</p>
  </div>
</body>
</html>`

  splashWindow.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(html)}`)
}

function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 960,
    minHeight: 640,
    show: false,
    title: '豪赋-虹膜颜色识别',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
    },
  })

  mainWindow.once('ready-to-show', () => {
    if (splashWindow && !splashWindow.isDestroyed()) {
      splashWindow.close()
      splashWindow = null
    }
    mainWindow.show()
  })

  mainWindow.on('closed', () => {
    mainWindow = null
  })

  mainWindow.loadURL(APP_URL)
}

async function startVisionService() {
  const pythonExe = getPythonExe()
  const visionDir = getVisionDir()
  const modelPath = path.join(visionDir, 'assets', 'models', 'face_landmarker.task')

  if (!fs.existsSync(visionDir)) {
    throw new Error(`未找到 iris-vision 目录: ${visionDir}`)
  }
  if (!fs.existsSync(modelPath)) {
    throw new Error(`未找到 MediaPipe 模型，请先运行 scripts/prepare-python.ps1 或 download_model.py`)
  }

  visionProcess = spawn(
    pythonExe,
    ['-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', String(VISION_PORT)],
    {
      cwd: visionDir,
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    },
  )

  visionProcess.stdout?.on('data', (d) => {
    if (!isPackaged()) console.log('[vision]', d.toString())
  })
  visionProcess.stderr?.on('data', (d) => {
    if (!isPackaged()) console.error('[vision]', d.toString())
  })
  visionProcess.on('exit', (code) => {
    if (code !== 0 && code !== null && mainWindow) {
      dialog.showErrorBox('Vision 服务异常退出', `iris-vision 进程已退出 (code=${code})`)
    }
  })

  await waitForHealth(VISION_HEALTH, 'iris-vision')
}

async function startApiService() {
  const javaExe = getJavaExe()
  const jarPath = getApiJar()

  if (!fs.existsSync(jarPath)) {
    throw new Error(
      `未找到 iris-api.jar: ${jarPath}\n请先执行 mvnw package -Pdesktop 并复制 JAR 到 iris-desktop/resources/`,
    )
  }

  apiProcess = spawn(
    javaExe,
    ['-jar', jarPath, '--spring.profiles.active=desktop'],
    {
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    },
  )

  apiProcess.stdout?.on('data', (d) => {
    if (!isPackaged()) console.log('[api]', d.toString())
  })
  apiProcess.stderr?.on('data', (d) => {
    if (!isPackaged()) console.error('[api]', d.toString())
  })
  apiProcess.on('exit', (code) => {
    if (code !== 0 && code !== null && mainWindow) {
      dialog.showErrorBox('API 服务异常退出', `iris-api 进程已退出 (code=${code})`)
    }
  })

  await waitForHealth(API_HEALTH, 'iris-api')
}

async function bootstrap() {
  createSplashWindow()

  const ports = [VISION_PORT, API_PORT]
  for (const port of ports) {
    if (await isPortInUse(port)) {
      throw new Error(`端口 ${port} 已被占用，请关闭占用该端口的程序后重试`)
    }
  }

  await startVisionService()
  await startApiService()
  createMainWindow()
}

function setupPermissions() {
  session.defaultSession.setPermissionRequestHandler((_wc, permission, callback) => {
    callback(permission === 'media' || permission === 'mediaKeySystem')
  })
  session.defaultSession.setPermissionCheckHandler((_wc, permission) => {
    return permission === 'media' || permission === 'mediaKeySystem'
  })
}

function shutdownServices() {
  killProcessTree(apiProcess)
  killProcessTree(visionProcess)
  apiProcess = null
  visionProcess = null
}

app.whenReady().then(async () => {
  setupPermissions()
  try {
    await bootstrap()
  } catch (err) {
    shutdownServices()
    if (splashWindow && !splashWindow.isDestroyed()) {
      splashWindow.close()
    }
    dialog.showErrorBox('启动失败', err instanceof Error ? err.message : String(err))
    app.quit()
  }
})

app.on('before-quit', () => {
  shutdownServices()
})

app.on('window-all-closed', () => {
  app.quit()
})

process.on('exit', shutdownServices)
