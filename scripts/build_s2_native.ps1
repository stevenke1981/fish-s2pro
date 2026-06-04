# Build mach92432/s2.cpp as a static library and link it into fish_s2_infer (Rust rewrite layer).
# Requires: CMake, Vulkan SDK, C++17, git
#
# Usage:
#   .\scripts\build_s2_native.ps1 -S2CppDir D:\src\s2.cpp
#   $env:S2_CPP_LIB = "D:\src\s2.cpp\build-native\lib"
#   cargo build -p fish_s2_gui --features cpp-engine

param(
    [Parameter(Mandatory = $true)]
    [string] $S2CppDir,
    [string] $BuildType = "Release"
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$ffiDir = Join-Path $root "crates\fish_s2_infer\ffi"
$buildDir = Join-Path $S2CppDir "build-native-rust"
$libDir = Join-Path $buildDir "lib"

if (-not (Test-Path $S2CppDir)) {
    Write-Error "S2CppDir not found: $S2CppDir"
}

Write-Host "Configuring s2.cpp CMake (Vulkan)..."
cmake -S $S2CppDir -B $buildDir -DCMAKE_BUILD_TYPE=$BuildType -DS2_VULKAN=ON
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Building ggml + s2 objects..."
cmake --build $buildDir --config $BuildType --parallel
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

New-Item -ItemType Directory -Force -Path $libDir | Out-Null

# Package our FFI + pipeline sources into static lib fish_s2_cpp (simplified: use ar/lib from cmake target if present)
$engineCpp = Join-Path $ffiDir "s2_engine_ffi.cpp"
$engineObj = Join-Path $buildDir "s2_engine_ffi.obj"

Write-Host "Compiling Rust FFI shim..."
$includes = @(
    "/I$S2CppDir\include",
    "/I$S2CppDir\ggml\include",
    "/I$S2CppDir\ggml\src",
    "/I$ffiDir"
)
cl.exe /std:c++17 /EHsc /c $includes /Fo:$engineObj $engineCpp
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$libOut = Join-Path $libDir "fish_s2_cpp.lib"
Write-Host "Creating $libOut (link s2 + ggml from cmake output — adjust paths if link fails)..."
# Users may need to add ggml.lib + s2 object libs from $buildDir manually on first integration.
lib.exe /OUT:$libOut $engineObj
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Set environment and rebuild Rust:"
Write-Host "  `$env:S2_CPP_LIB = `"$libDir`""
Write-Host "  `$env:S2_CPP_DIR = `"$libDir`""
Write-Host "  cargo build -p fish_s2_gui --features cpp-engine"