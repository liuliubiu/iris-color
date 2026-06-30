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
$WebDistJs = Get-ChildItem (Join-Path $RootDir "iris-web\dist\assets\*.js") -ErrorAction SilentlyContinue | Select-Object -First 1

$env:CSC_IDENTITY_AUTO_DISCOVERY = "false"

Push-Location $DesktopDir
try {
    $SyncBrand = Join-Path $RootDir "scripts\sync-brand.ps1"
    if (Test-Path $SyncBrand) {
        Write-Host "=== sync brand icons ==="
        & $SyncBrand
    }

    if (-not $SkipRuntimePrep) {
        $JreExe = Join-Path $ResourcesDir "jre\bin\java.exe"
        $PyExe = Join-Path $ResourcesDir "python\python.exe"
        if (-not (Test-Path $JreExe)) {
            Write-Host "=== prepare JRE ==="
            & (Join-Path $PSScriptRoot "prepare-jre.ps1")
        }
        if (-not (Test-Path $PyExe)) {
            Write-Host "=== prepare Python ==="
            & (Join-Path $PSScriptRoot "prepare-python.ps1")
        }
    }

    if ($SkipMaven) {
        Write-Warning "SkipMaven: resources/iris-api.jar will NOT be rebuilt. Use only for icon-only retries."
    } else {
        Write-Host "=== build iris-api JAR (with frontend static) ==="
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

    $WebDistJs = Get-ChildItem (Join-Path $RootDir "iris-web\dist\assets\*.js") -ErrorAction SilentlyContinue | Select-Object -First 1

    if ($WebDistJs) {
        $jarList = jar tf (Join-Path $ResourcesDir "iris-api.jar") 2>$null
        $bundleName = $WebDistJs.Name
        if ($jarList -match [regex]::Escape($bundleName)) {
            Write-Host "JAR contains frontend bundle: $bundleName"
        } else {
            throw "JAR missing frontend bundle $bundleName - run without -SkipMaven"
        }
    }

    $BuildInfo = @(
        "version=$((Get-Content (Join-Path $DesktopDir 'package.json') | ConvertFrom-Json).version)"
        "built_at=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        "frontend=$($WebDistJs.Name)"
    ) -join "`n"
    Set-Content -Path (Join-Path $ResourcesDir "BUILD_INFO.txt") -Value $BuildInfo -Encoding UTF8

    Write-Host "=== npm install (electron) ==="
    npm install
    if ($LASTEXITCODE -ne 0) { throw "npm install failed" }

    if (-not $SkipDist) {
        Write-Host "=== pack Windows installer ==="
        if (Test-Path (Join-Path $DesktopDir "dist")) {
            Remove-Item (Join-Path $DesktopDir "dist\*") -Recurse -Force -ErrorAction SilentlyContinue
        }
        npm run dist
        if ($LASTEXITCODE -ne 0) { throw "electron-builder failed" }
        Write-Host ""
        Write-Host "Done. Installer in: $(Join-Path $DesktopDir 'dist')"
        Write-Host "IMPORTANT: Uninstall old 'IrisColor' if present, then install the new Setup exe."
    }
}
finally {
    Pop-Location
}
