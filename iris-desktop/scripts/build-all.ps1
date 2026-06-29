#Requires -Version 5.1
<#
.SYNOPSIS
  一键构建 IrisColor Windows 安装包
#>
param(
    [switch]$SkipRuntimePrep,
    [switch]$SkipDist,
    [switch]$SkipMaven
)

. (Join-Path $PSScriptRoot "_init.ps1")

$ErrorActionPreference = "Stop"
$DesktopDir = Join-Path $PSScriptRoot ".."
$RootDir = Join-Path $DesktopDir ".."
$ApiDir = Join-Path $RootDir "iris-api"
$ResourcesDir = Join-Path $DesktopDir "resources"

# 禁用 electron-builder 自动代码签名（避免下载 winCodeSign 及符号链接权限问题）
$env:CSC_IDENTITY_AUTO_DISCOVERY = "false"

Push-Location $DesktopDir
try {
    $SyncBrand = Join-Path $RootDir "scripts\sync-brand.ps1"
    if (Test-Path $SyncBrand) {
        Write-Host "=== 同步品牌图标 ==="
        & $SyncBrand
    }

    if (-not $SkipRuntimePrep) {
        $JreExe = Join-Path $ResourcesDir "jre\bin\java.exe"
        $PyExe = Join-Path $ResourcesDir "python\python.exe"
        if (-not (Test-Path $JreExe)) {
            Write-Host "=== 准备 JRE ==="
            & (Join-Path $PSScriptRoot "prepare-jre.ps1")
        }
        if (-not (Test-Path $PyExe)) {
            Write-Host "=== 准备 Python ==="
            & (Join-Path $PSScriptRoot "prepare-python.ps1")
        }
    }

    if (-not $SkipMaven) {
        Write-Host "=== 构建 iris-api JAR（含前端 static）==="
        Push-Location $ApiDir
        & .\mvnw.cmd clean package -Pdesktop -DskipTests
        if ($LASTEXITCODE -ne 0) { throw "Maven build failed" }
        Pop-Location

        $Jar = Get-ChildItem -Path (Join-Path $ApiDir "target") -Filter "iris-api-*.jar" |
            Where-Object { $_.Name -notmatch "original" } |
            Select-Object -First 1
        if (-not $Jar) {
            throw "iris-api-*.jar not found in target/"
        }

        New-Item -ItemType Directory -Path $ResourcesDir -Force | Out-Null
        Copy-Item -Path $Jar.FullName -Destination (Join-Path $ResourcesDir "iris-api.jar") -Force
        Write-Host "Copied JAR: $($Jar.FullName)"
    }

    Write-Host "=== 安装 Electron 依赖 ==="
    if (-not (Test-Path (Join-Path $DesktopDir "node_modules"))) {
        npm install
        if ($LASTEXITCODE -ne 0) { throw "npm install failed" }
    }

    if (-not $SkipDist) {
        Write-Host "=== 打包 Windows 安装程序 ==="
        npm run dist
        if ($LASTEXITCODE -ne 0) { throw "electron-builder failed" }
        Write-Host ""
        Write-Host "Done. Installer: $(Join-Path $DesktopDir 'dist')"
    }
}
finally {
    Pop-Location
}
