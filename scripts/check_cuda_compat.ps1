param(
    [int] $CudaDevice = 0,
    [string] $CudaArchitectures,
    [string] $S2CppDir = (Join-Path (Split-Path $PSScriptRoot -Parent) "output\s2.cpp-src"),
    [string] $Transformer = (Join-Path (Split-Path $PSScriptRoot -Parent) "models\s2-pro-f16-transformer-only.gguf"),
    [string] $Report = (Join-Path (Split-Path $PSScriptRoot -Parent) "output\cuda_compat_report.json"),
    [switch] $RunBuildSmoke,
    [switch] $AllowUnsupportedCudaCompiler,
    [switch] $RequireCuda
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Use-UnicodeEncoding.ps1")

function Get-CommandPath {
    param([string] $Name)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Invoke-Capture {
    param(
        [string] $FilePath,
        [string[]] $Arguments
    )
    if (-not $FilePath) {
        return [ordered]@{
            exit_code = $null
            stdout = ""
            stderr = ""
        }
    }

    $stdout = New-TemporaryFile
    $stderr = New-TemporaryFile
    try {
        $process = Start-Process `
            -FilePath $FilePath `
            -ArgumentList $Arguments `
            -NoNewWindow `
            -PassThru `
            -Wait `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr
        return [ordered]@{
            exit_code = $process.ExitCode
            stdout = (Get-Content -Raw -LiteralPath $stdout -ErrorAction SilentlyContinue)
            stderr = (Get-Content -Raw -LiteralPath $stderr -ErrorAction SilentlyContinue)
        }
    } finally {
        Remove-Item -LiteralPath $stdout, $stderr -ErrorAction SilentlyContinue
    }
}

function Parse-NvidiaSmiGpuRows {
    param([string] $Text)

    $rows = @()
    foreach ($line in ($Text -split "`r?`n")) {
        $trimmed = $line.Trim()
        if (-not $trimmed) { continue }
        $parts = $trimmed -split "\s*,\s*"
        if ($parts.Count -lt 5) { continue }
        $rows += [ordered]@{
            index = [int] $parts[0]
            name = $parts[1]
            compute_capability = $parts[2]
            driver_version = $parts[3]
            memory_total = $parts[4]
        }
    }
    return @($rows)
}

function Convert-ComputeCapabilityToArch {
    param([string] $ComputeCapability)

    if ($ComputeCapability -match '^(\d+)\.(\d+)$') {
        return "$($matches[1])$($matches[2])"
    }
    return $null
}

function Parse-NvccVersion {
    param([string] $Text)

    if ($Text -match 'release\s+([0-9]+(?:\.[0-9]+)*)') {
        return $matches[1]
    }
    return $null
}

$root = Split-Path $PSScriptRoot -Parent
$nvidiaSmi = Get-CommandPath "nvidia-smi"
$nvcc = Get-CommandPath "nvcc"
$cmake = Get-CommandPath "cmake"
$cl = Get-CommandPath "cl"

$nvidiaQuery = Invoke-Capture `
    -FilePath $nvidiaSmi `
    -Arguments @(
        "--query-gpu=index,name,compute_cap,driver_version,memory.total",
        "--format=csv,noheader,nounits"
    )
$gpus = if ($nvidiaQuery.exit_code -eq 0) { Parse-NvidiaSmiGpuRows $nvidiaQuery.stdout } else { @() }
$selectedGpu = $gpus | Where-Object { $_.index -eq $CudaDevice } | Select-Object -First 1
$detectedArch = if ($selectedGpu) {
    Convert-ComputeCapabilityToArch $selectedGpu.compute_capability
} else {
    $null
}
$effectiveArch = if ($CudaArchitectures) { $CudaArchitectures } elseif ($detectedArch) { $detectedArch } else { "86" }

$nvccVersion = $null
if ($nvcc) {
    $nvccQuery = Invoke-Capture -FilePath $nvcc -Arguments @("--version")
    if ($nvccQuery.exit_code -eq 0) {
        $nvccVersion = Parse-NvccVersion ($nvccQuery.stdout + "`n" + $nvccQuery.stderr)
    }
} else {
    $nvccQuery = [ordered]@{ exit_code = $null; stdout = ""; stderr = "" }
}

$checks = [ordered]@{
    nvidia_smi_found = [bool] $nvidiaSmi
    nvidia_smi_path = $nvidiaSmi
    nvidia_smi_ok = ($nvidiaQuery.exit_code -eq 0)
    nvcc_found = [bool] $nvcc
    nvcc_path = $nvcc
    nvcc_ok = ($nvccQuery.exit_code -eq 0)
    nvcc_version = $nvccVersion
    cmake_found = [bool] $cmake
    cmake_path = $cmake
    msvc_cl_found = [bool] $cl
    msvc_cl_path = $cl
    selected_device_found = [bool] $selectedGpu
}

$recommendations = [System.Collections.Generic.List[string]]::new()
if (-not $checks.nvidia_smi_found) {
    $recommendations.Add("Install NVIDIA driver and ensure nvidia-smi is on PATH.") | Out-Null
}
if (-not $checks.nvidia_smi_ok -and $checks.nvidia_smi_found) {
    $recommendations.Add("nvidia-smi was found but did not return GPU info; check driver/runtime visibility.") | Out-Null
}
if (-not $checks.nvcc_found) {
    $recommendations.Add("Install CUDA Toolkit and ensure nvcc is on PATH for GGML_CUDA builds.") | Out-Null
}
if (-not $checks.cmake_found) {
    $recommendations.Add("Install CMake and ensure cmake is on PATH.") | Out-Null
}
if (-not $selectedGpu) {
    $recommendations.Add("Selected CUDA device $CudaDevice was not reported by nvidia-smi.") | Out-Null
}
if (-not $CudaArchitectures -and $detectedArch) {
    $recommendations.Add("Use -CudaArchitectures $detectedArch for this GPU, or leave unset to auto-select it.") | Out-Null
}

$buildSmoke = [ordered]@{
    requested = [bool] $RunBuildSmoke
    status = if ($RunBuildSmoke) { "pending" } else { "skipped" }
    exit_code = $null
    output_tail = @()
}

if ($RunBuildSmoke) {
    $buildScript = Join-Path $PSScriptRoot "dump_s2cpp_slow_ar_stats.ps1"
    if (-not (Test-Path -LiteralPath $buildScript)) {
        throw "-RunBuildSmoke requires source-tree scripts/dump_s2cpp_slow_ar_stats.ps1"
    }
    if (-not (Test-Path -LiteralPath $S2CppDir)) {
        throw "S2CppDir not found: $S2CppDir"
    }
    if (-not (Test-Path -LiteralPath $Transformer)) {
        throw "Transformer not found: $Transformer"
    }
    $buildArgs = @{
        S2CppDir = $S2CppDir
        Transformer = $Transformer
        Cuda = $true
        CudaDevice = $CudaDevice
        CudaArchitectures = $effectiveArch
        BuildOnly = $true
    }
    if ($AllowUnsupportedCudaCompiler) {
        $buildArgs["AllowUnsupportedCudaCompiler"] = $true
    }
    $smokeOutput = & $buildScript @buildArgs 2>&1
    $buildSmoke.exit_code = if ($null -ne $LASTEXITCODE) { [int] $LASTEXITCODE } else { 0 }
    if ($smokeOutput.Count -gt 0) {
        $tailStart = [Math]::Max(0, $smokeOutput.Count - 80)
        $buildSmoke.output_tail = @($smokeOutput[$tailStart..($smokeOutput.Count - 1)] | ForEach-Object { $_.ToString() })
    }
    $buildSmoke.status = if ($buildSmoke.exit_code -eq 0) { "passed" } else { "failed" }
    if ($buildSmoke.exit_code -ne 0) {
        throw "CUDA build smoke failed with exit code $($buildSmoke.exit_code)"
    }
}

$cudaReady = (
    $checks.nvidia_smi_ok -and
    $checks.nvcc_ok -and
    $checks.cmake_found -and
    $checks.selected_device_found
)

$reportObject = [ordered]@{
    schema = "fish-s2pro.cuda-compat.v1"
    generated_at = (Get-Date).ToUniversalTime().ToString("o")
    root = $root
    cuda_ready = [bool] $cudaReady
    cuda_device = $CudaDevice
    cuda_architectures = $effectiveArch
    detected_architectures = $detectedArch
    gpus = @($gpus)
    checks = $checks
    build_smoke = $buildSmoke
    recommendations = @($recommendations)
}

Write-Utf8NoBom $Report (($reportObject | ConvertTo-Json -Depth 8) + "`n")
Write-Host "cuda_ready=$([bool] $cudaReady)"
Write-Host "cuda_architectures=$effectiveArch"
Write-Host "report=$Report"

if ($RequireCuda -and -not $cudaReady) {
    exit 1
}
