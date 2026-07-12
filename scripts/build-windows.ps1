# scripts/build-windows.ps1 — Build do BRBtcHuntAMD-OpenCL no Windows
#
# Pré-requisitos:
#   - Visual Studio 2022 (com workload "Desktop development with C++")
#   - CMake 3.16+ (choco install cmake)
#   - AMD Adrenalin driver (inclui OpenCL runtime)
#   - AMD APP SDK (para headers OpenCL) OU choco install opencl-headers
#   - OpenSSL (vcpkg install openssl:x64-windows OU choco install openssl)
#
# Uso:
#   cd C:\path\to\BRBtcHuntAMD-OpenCL
#   powershell -ExecutionPolicy Bypass -File scripts\build-windows.ps1
#
# Após o build, o executável estará em build\Release\BRBtcHuntAMD-OpenCL.exe
# e a pasta kernels/ será copiada para o mesmo diretório.

param(
    [string]$BuildType = "Release",
    [string]$OpenSSLRoot = $env:OPENSSL_ROOT_DIR
)

$ErrorActionPreference = "Stop"

Write-Host "=== BRBtcHuntAMD-OpenCL Build Script ===" -ForegroundColor Cyan
Write-Host ""

# ── 1. Verifica pré-requisitos ────────────────────────────────────────
Write-Host "[1/5] Checking prerequisites..." -ForegroundColor Yellow

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Write-Error "CMake not found. Install: choco install cmake"
    exit 1
}

if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    # Tenta carregar ambiente do VS
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($vsPath) {
            $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path $vcvars) {
                Write-Host "  Loading MSVC environment from $vsPath"
                cmd /c "`"$vcvars`" && set" | ForEach-Object {
                    if ($_ -match "^(.*?)=(.*)$") {
                        Set-Item -Path "env:$($matches[1])" -Value $matches[2]
                    }
                }
            }
        }
    }
}

if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    Write-Error "MSVC compiler (cl.exe) not found. Install Visual Studio with C++ workload."
    exit 1
}

# Detecta OpenCL
$env:AMDAPPSDKROOT = $env:AMDAPPSDKROOT
if (-not $env:AMDAPPSDKROOT) {
    $oclHeader = "${env:ProgramFiles}\AMD APP SDK\3.0\include\CL\opencl.h"
    if (Test-Path $oclHeader) {
        $env:AMDAPPSDKROOT = "${env:ProgramFiles}\AMD APP SDK\3.0"
        Write-Host "  Found AMD APP SDK at $env:AMDAPPSDKROOT"
    }
}
if (-not $env:AMDAPPSDKROOT) {
    Write-Warning "AMDAPPSDKROOT not set. Install AMD APP SDK or set OPENCL_INCLUDE_DIR manually."
}

# OpenSSL
if (-not $OpenSSLRoot) {
    $candidate = "C:\vcpkg\packages\openssl_x64-windows"
    if (Test-Path $candidate) { $OpenSSLRoot = $candidate }
}
if ($OpenSSLRoot) {
    $env:OPENSSL_ROOT_DIR = $OpenSSLRoot
    Write-Host "  OpenSSL root: $OpenSSLRoot"
}

Write-Host "  OK" -ForegroundColor Green
Write-Host ""

# ── 2. Cria diretório de build ────────────────────────────────────────
Write-Host "[2/5] Creating build directory..." -ForegroundColor Yellow

$buildDir = "build"
if (Test-Path $buildDir) {
    Remove-Item -Recurse -Force $buildDir
}
New-Item -ItemType Directory -Path $buildDir | Out-Null
Write-Host ""

# ── 3. CMake configure ────────────────────────────────────────────────
Write-Host "[3/5] Running CMake configure..." -ForegroundColor Yellow

$cmakeArgs = @(
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DCMAKE_C_COMPILER=cl",
    "-DCMAKE_CXX_COMPILER=cl",
    "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
)

if ($env:AMDAPPSDKROOT) {
    $cmakeArgs += "-DOPENCL_INCLUDE_DIR=$env:AMDAPPSDKROOT\include"
    $cmakeArgs += "-DOPENCL_LIBRARY=$env:AMDAPPSDKROOT\lib\x86_64\OpenCL.lib"
}

& cmake -S . -B $buildDir @cmakeArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configuration failed."
    exit 1
}
Write-Host ""

# ── 4. Build ──────────────────────────────────────────────────────────
Write-Host "[4/5] Building..." -ForegroundColor Yellow

& cmake --build $buildDir --config $BuildType --parallel
if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed."
    exit 1
}
Write-Host ""

# ── 5. Finalização ────────────────────────────────────────────────────
Write-Host "[5/5] Build complete!" -ForegroundColor Green
Write-Host ""

$exe = Join-Path $buildDir "BRBtcHuntAMD-OpenCL.exe"
if (Test-Path $exe) {
    Write-Host "Executable: $exe" -ForegroundColor Cyan
} else {
    Write-Host "Executable: $buildDir\BRBtcHuntAMD-OpenCL.exe (or check $buildDir\$BuildType\)" -ForegroundColor Cyan
}
Write-Host "Kernels:    $buildDir\kernels\" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run:" -ForegroundColor White
Write-Host "  cd $buildDir" -ForegroundColor White
Write-Host "  .\BRBtcHuntAMD-OpenCL.exe --range 200000000:3FFFFFFFF --address 1HBtApAwR... --grid 128,256 --slices 64" -ForegroundColor White
Write-Host ""
Write-Host "For GPU detection only (no search):" -ForegroundColor White
Write-Host "  .\BRBtcHuntAMD-OpenCL.exe --help" -ForegroundColor White
