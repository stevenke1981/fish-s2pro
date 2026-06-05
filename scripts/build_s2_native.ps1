# Build mach92432/s2.cpp as a static library and link it into fish_s2_infer (Rust rewrite layer).
# Requires: CMake, C++17, git. Add -UseCuda for GGML_CUDA builds.
#
# Usage:
#   .\scripts\build_s2_native.ps1 -S2CppDir D:\src\s2.cpp
#   .\scripts\build_s2_native.ps1 -S2CppDir D:\src\s2.cpp -UseCuda -CudaArchitectures 86
#   $env:S2_CPP_LIB = "D:\src\s2.cpp\build-native\lib"
#   cargo build -p fish_s2_gui --features cpp-engine

param(
    [Parameter(Mandatory = $true)]
    [string] $S2CppDir,
    [string] $BuildType = "Release",
    [switch] $UseCuda,
    [int] $CudaDevice = 0,
    [string] $CudaArchitectures = "86",
    [switch] $AllowUnsupportedCudaCompiler
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$ffiDir = Join-Path $root "crates\fish_s2_infer\ffi"
$buildDirName = if ($UseCuda) { "build-native-rust-cuda" } else { "build-native-rust" }
$buildDir = Join-Path $S2CppDir $buildDirName
$ggmlBuildDir = Join-Path $buildDir "ggml"
$objDir = Join-Path $buildDir "obj"
$libDir = Join-Path $buildDir "lib"

if (-not (Test-Path $S2CppDir)) {
    Write-Error "S2CppDir not found: $S2CppDir"
}

function Patch-S2CppCudaHooks {
    param([string] $SourceRoot)

    $codecCpp = Join-Path $SourceRoot "src\s2_codec.cpp"
    if (Test-Path -LiteralPath $codecCpp) {
        $raw = Get-Content -Raw -LiteralPath $codecCpp
        if (-not $raw.Contains("ggml-cuda.h")) {
            Write-Host "Patching s2_codec.cpp CUDA include..."
            $raw = [regex]::Replace(
                $raw,
                '(#ifdef GGML_USE_VULKAN\s*#include "ggml-vulkan.h"\s*#endif)',
                '$1' + "`n#ifdef GGML_USE_CUDA`n#include `"ggml-cuda.h`"`n#endif",
                1
            )
        }
        if (-not $raw.Contains("FISH_S2_CUDA_DEVICE")) {
            Write-Host "Patching s2_codec.cpp CUDA backend hook..."
            $cudaHook = @'
#ifdef GGML_USE_CUDA
    if (!impl_->backend) {
        if (const char * cuda_device_env = std::getenv("FISH_S2_CUDA_DEVICE")) {
            const int cuda_device = std::atoi(cuda_device_env);
            impl_->backend = ggml_backend_cuda_init(cuda_device);
            if (impl_->backend) {
                std::cout << "[Codec] CUDA backend initialized on device " << cuda_device << "." << std::endl;
            } else {
                std::cerr << "[Codec] CUDA init failed for device " << cuda_device << ", falling back to CPU." << std::endl;
            }
        }
    }
#endif
'@
            $raw = $raw.Replace(
                "    if (!impl_->backend) impl_->backend = ggml_backend_cpu_init();",
                $cudaHook + "    if (!impl_->backend) impl_->backend = ggml_backend_cpu_init();"
            )
        }
        if (-not $raw.Contains("#include <cstdlib>")) {
            $raw = $raw.Replace("#include <stdexcept>", "#include <stdexcept>`n#include <cstdlib>")
        }
        [System.IO.File]::WriteAllText($codecCpp, $raw, [System.Text.UTF8Encoding]::new($false))
    }

    $pipelineCpp = Join-Path $SourceRoot "src\s2_pipeline.cpp"
    if (Test-Path -LiteralPath $pipelineCpp) {
        $raw = Get-Content -Raw -LiteralPath $pipelineCpp
        $changed = $false
        $old = 'Codec loaded on CPU (fallback).'
        if ($raw.Contains($old)) {
            Write-Host "Patching s2_pipeline.cpp fallback backend log..."
            $raw = $raw.Replace($old, 'Codec loaded on fallback backend.')
            $changed = $true
        }
        $vulkanFallbackLog = 'Pipeline warning: codec failed on GPU, falling back to CPU.'
        if ($raw.Contains($vulkanFallbackLog)) {
            Write-Host "Patching s2_pipeline.cpp Vulkan fallback wording..."
            $raw = $raw.Replace($vulkanFallbackLog, 'Pipeline warning: codec Vulkan path unavailable; using configured fallback backend.')
            $changed = $true
        }
        $wordingPatches = @(
            @('GPU assignment: model -> GPU ', 'Vulkan device assignment: model -> '),
            @(', codec -> GPU ', ', codec -> '),
            @('Loading model on GPU ', 'Loading model with Vulkan device '),
            @('Model loaded on GPU ', 'Model loaded with Vulkan device '),
            @('Loading codec on GPU ', 'Loading codec with Vulkan device ')
        )
        foreach ($patch in $wordingPatches) {
            if ($raw.Contains($patch[0])) {
                Write-Host "Patching s2_pipeline.cpp Vulkan/CUDA device wording..."
                $raw = $raw.Replace($patch[0], $patch[1])
                $changed = $true
            }
        }
        $textLog = 'std::cout << "Text: " << params.text << std::endl;'
        if ($raw.Contains($textLog)) {
            Write-Host "Patching s2_pipeline.cpp UTF-8 text log..."
            $raw = $raw.Replace($textLog, 'std::cout << "Text bytes: " << params.text.size() << std::endl;')
            $changed = $true
        }
        if ($changed) {
            [System.IO.File]::WriteAllText($pipelineCpp, $raw, [System.Text.UTF8Encoding]::new($false))
        }
    }
}

if ($UseCuda) {
    Patch-S2CppCudaHooks $S2CppDir

    $cmakeLists = Join-Path $S2CppDir "CMakeLists.txt"
    if (Test-Path -LiteralPath $cmakeLists) {
        $raw = Get-Content -Raw -LiteralPath $cmakeLists
        $forcedVulkan = 'set(GGML_VULKAN ON CACHE BOOL "" FORCE)'
        if ($raw.Contains($forcedVulkan)) {
            Write-Host "Patching s2.cpp CMakeLists.txt to avoid forced Vulkan for CUDA builds..."
            $raw = $raw.Replace($forcedVulkan, 'set(GGML_VULKAN ${S2_VULKAN} CACHE BOOL "" FORCE)')
            [System.IO.File]::WriteAllText($cmakeLists, $raw, [System.Text.UTF8Encoding]::new($false))
        }
    }
}

Write-Host "Configuring ggml CMake..."
$cmakeArgs = @(
    "-S", (Join-Path $S2CppDir "ggml"),
    "-B", $ggmlBuildDir,
    "-DCMAKE_BUILD_TYPE=$BuildType"
)
if ($UseCuda) {
    $cmakeArgs += @(
        "-DGGML_CUDA=ON",
        "-DS2_VULKAN=OFF",
        "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures"
    )
    if ($AllowUnsupportedCudaCompiler) {
        $cmakeArgs += "-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler"
    }
} else {
    $cmakeArgs += @("-DGGML_CUDA=OFF", "-DGGML_VULKAN=ON")
}
cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Building ggml..."
cmake --build $ggmlBuildDir --config $BuildType --parallel
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

New-Item -ItemType Directory -Force -Path $libDir | Out-Null
New-Item -ItemType Directory -Force -Path $objDir | Out-Null

Write-Host "Compiling s2 pipeline + Rust FFI shim..."
$includes = @(
    "/I$S2CppDir\include",
    "/I$S2CppDir\third_party",
    "/I$S2CppDir\ggml\include",
    "/I$S2CppDir\ggml\src",
    "/I$ffiDir"
)
$defines = @()
if ($UseCuda) {
    $defines += "/DGGML_USE_CUDA"
} else {
    $defines += "/DGGML_USE_VULKAN"
}
$sources = @(
    "src\s2_audio.cpp",
    "src\s2_tokenizer.cpp",
    "src\s2_sampler.cpp",
    "src\s2_model.cpp",
    "src\s2_codec.cpp",
    "src\s2_prompt.cpp",
    "src\s2_generate.cpp",
    "src\s2_pipeline.cpp"
) | ForEach-Object { Join-Path $S2CppDir $_ }
$sources += Join-Path $ffiDir "s2_engine_ffi.cpp"
$objects = @()
foreach ($source in $sources) {
    $obj = Join-Path $objDir (([System.IO.Path]::GetFileNameWithoutExtension($source)) + ".obj")
    if ($source.EndsWith("s2_engine_ffi.cpp")) {
        $obj = Join-Path $objDir "s2_engine_ffi.obj"
    }
    cl.exe /std:c++17 /EHsc /c $includes $defines "/Fo$obj" $source
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $objects += $obj
}

$libOut = Join-Path $libDir "fish_s2_cpp.lib"
Write-Host "Creating $libOut..."
$staticLibs = @(
    Get-ChildItem -LiteralPath $ggmlBuildDir -Recurse -Filter "*.lib" -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -ne $libOut } |
        ForEach-Object { $_.FullName }
)
lib.exe /OUT:$libOut @objects @staticLibs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
if ($UseCuda) {
    Write-Host "CUDA backend requested:"
    Write-Host "  FISH_S2_CUDA_DEVICE=$CudaDevice"
    Write-Host "  CMAKE_CUDA_ARCHITECTURES=$CudaArchitectures"
}
Write-Host "Set environment and rebuild Rust:"
Write-Host "  `$env:S2_CPP_LIB = `"$libDir`""
Write-Host "  `$env:S2_CPP_DIR = `"$libDir`""
Write-Host "  cargo build -p fish_s2_gui --features cpp-engine"
