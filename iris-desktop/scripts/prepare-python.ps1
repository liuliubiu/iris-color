#Requires -Version 5.1
<#
.SYNOPSIS
  下载 Windows Embeddable Python 3.11，安装依赖到 iris-desktop/resources/python/
#>
param(
    [switch]$Force,
    [string]$PythonVersion = "3.11.9"
)

. (Join-Path $PSScriptRoot "_init.ps1")

$ErrorActionPreference = "Stop"
$RootDir = Join-Path $PSScriptRoot ".."
$ResourcesDir = Join-Path $RootDir "resources"
$PythonDir = Join-Path $ResourcesDir "python"
$PythonExe = Join-Path $PythonDir "python.exe"
$VisionDir = Join-Path $RootDir "..\iris-vision"
$Requirements = Join-Path $VisionDir "requirements.txt"
$ModelPath = Join-Path $VisionDir "assets\models\face_landmarker.task"

if ((Test-Path $PythonExe) -and -not $Force) {
    Write-Host "Python 已存在: $PythonExe"
    exit 0
}

if (-not (Test-Path $Requirements)) {
    throw "未找到 requirements.txt: $Requirements"
}

Write-Host "正在下载 Python $PythonVersion embeddable ..."
$ZipName = "python-$PythonVersion-embed-amd64.zip"
$ZipUrl = "https://www.python.org/ftp/python/$PythonVersion/$ZipName"
$ZipPath = Join-Path $env:TEMP $ZipName

Invoke-WebRequest -Uri $ZipUrl -OutFile $ZipPath -UseBasicParsing

if (Test-Path $PythonDir) {
    Remove-Item $PythonDir -Recurse -Force
}
New-Item -ItemType Directory -Path $PythonDir -Force | Out-Null
Expand-Archive -Path $ZipPath -DestinationPath $PythonDir -Force
Remove-Item $ZipPath -Force -ErrorAction SilentlyContinue

# 启用 site-packages
$PthFile = Get-ChildItem -Path $PythonDir -Filter "python*._pth" | Select-Object -First 1
if (-not $PthFile) {
    throw "未找到 python*._pth"
}
$PthContent = @(
    "python311.zip",
    ".",
    "Lib\site-packages",
    "",
    "# Enable site",
    "import site"
)
Set-Content -Path $PthFile.FullName -Value $PthContent -Encoding ASCII

$SitePackages = Join-Path $PythonDir "Lib\site-packages"
New-Item -ItemType Directory -Path $SitePackages -Force | Out-Null

Write-Host "正在安装 pip ..."
$GetPipUrl = "https://bootstrap.pypa.io/get-pip.py"
$GetPipPath = Join-Path $env:TEMP "get-pip.py"
Invoke-WebRequest -Uri $GetPipUrl -OutFile $GetPipPath -UseBasicParsing
& $PythonExe $GetPipPath --no-warn-script-location
Remove-Item $GetPipPath -Force -ErrorAction SilentlyContinue

Write-Host "正在安装 iris-vision 依赖（可能需要数分钟）..."
& $PythonExe -m pip install `
    -r $Requirements `
    --target $SitePackages `
    --no-warn-script-location `
    --upgrade

Write-Host "正在下载 MediaPipe 模型 ..."
if (-not (Test-Path $ModelPath)) {
    & $PythonExe (Join-Path $VisionDir "scripts\download_model.py")
}

if (-not (Test-Path $ModelPath)) {
    throw "模型下载失败: $ModelPath"
}

Write-Host "验证 uvicorn ..."
& $PythonExe -c "import uvicorn, fastapi, cv2, mediapipe; print('ok')"
if ($LASTEXITCODE -ne 0) {
    throw "Python 依赖验证失败"
}

Write-Host "Python 运行时已就绪: $PythonExe"
