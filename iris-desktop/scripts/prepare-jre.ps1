#Requires -Version 5.1
<#
.SYNOPSIS
  下载 Eclipse Temurin JRE 17 到 iris-desktop/resources/jre/
#>
param(
    [switch]$Force
)

. (Join-Path $PSScriptRoot "_init.ps1")

$ErrorActionPreference = "Stop"
$ResourcesDir = Join-Path $PSScriptRoot "..\resources"
$JreDir = Join-Path $ResourcesDir "jre"
$JavaExe = Join-Path $JreDir "bin\java.exe"

if ((Test-Path $JavaExe) -and -not $Force) {
    Write-Host "JRE 已存在: $JavaExe"
    exit 0
}

Write-Host "正在下载 Temurin JRE 17 (Windows x64) ..."
$ApiUrl = "https://api.adoptium.net/v3/binary/latest/17/ga/windows/x64/jre/hotspot/normal/eclipse?project=jdk"
$ZipPath = Join-Path $env:TEMP "temurin-jre17.zip"
$ExtractRoot = Join-Path $env:TEMP "temurin-jre17-extract"

Invoke-WebRequest -Uri $ApiUrl -OutFile $ZipPath -UseBasicParsing

if (Test-Path $ExtractRoot) {
    Remove-Item $ExtractRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $ExtractRoot -Force | Out-Null
Expand-Archive -Path $ZipPath -DestinationPath $ExtractRoot -Force

$InnerDir = Get-ChildItem -Path $ExtractRoot -Directory | Select-Object -First 1
if (-not $InnerDir) {
    throw "解压 JRE 失败：未找到目录"
}

if (Test-Path $JreDir) {
    Remove-Item $JreDir -Recurse -Force
}
New-Item -ItemType Directory -Path $ResourcesDir -Force | Out-Null
Move-Item -Path $InnerDir.FullName -Destination $JreDir

Remove-Item $ZipPath -Force -ErrorAction SilentlyContinue
Remove-Item $ExtractRoot -Recurse -Force -ErrorAction SilentlyContinue

if (-not (Test-Path $JavaExe)) {
    throw "JRE 安装失败：未找到 $JavaExe"
}

Write-Host "JRE 已就绪: $JavaExe"
& $JavaExe -version
