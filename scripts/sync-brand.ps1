#Requires -Version 5.1
<#
.SYNOPSIS
  Sync brand icons from brand/ to web public and desktop build dirs
#>
param(
    [string]$BrandDir = (Join-Path $PSScriptRoot "..\brand")
)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
$WebBrand = Join-Path $Root "iris-web\public\brand"
$DesktopBuild = Join-Path $Root "iris-desktop\build"

if (-not (Test-Path $BrandDir)) {
    Write-Warning "brand directory not found: $BrandDir"
    exit 0
}

New-Item -ItemType Directory -Path $WebBrand -Force | Out-Null
New-Item -ItemType Directory -Path $DesktopBuild -Force | Out-Null

function Sync-File {
    param([string]$Name, [string]$DestDir, [string]$DestName = $Name)
    $Src = Join-Path $BrandDir $Name
    if (-not (Test-Path $Src)) {
        Write-Host "  skip (missing): $Name"
        return
    }
    $Dest = Join-Path $DestDir $DestName
    Copy-Item -Path $Src -Destination $Dest -Force
    Write-Host "  synced: $Name -> $Dest"
}

Write-Host "=== sync brand icons ==="
Sync-File "logo.png" $WebBrand
Sync-File "favicon.ico" $WebBrand
Sync-File "favicon.png" $WebBrand
Sync-File "apple-touch-icon.png" $WebBrand
Sync-File "logo.png" $DesktopBuild
Sync-File "app-icon.ico" $DesktopBuild "icon.ico"
Write-Host "done."
