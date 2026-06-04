# Shared UTF-8 (Unicode) console and file I/O for fish-s2pro scripts on Windows.
# Dot-source: . (Join-Path $PSScriptRoot 'Use-UnicodeEncoding.ps1')

$script:Utf8NoBom = [System.Text.UTF8Encoding]::new($false)

function Initialize-UnicodeConsole {
    $script:Utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    [Console]::InputEncoding = $script:Utf8NoBom
    [Console]::OutputEncoding = $script:Utf8NoBom
    $global:OutputEncoding = $script:Utf8NoBom
    if ($IsWindows -or $env:OS -match 'Windows') {
        try { chcp 65001 | Out-Null } catch { }
    }
}

function Read-Utf8 {
    param([string] $Path)
    return [System.IO.File]::ReadAllText($Path, $script:Utf8NoBom)
}

function Write-Utf8NoBom {
    param([string] $Path, [string] $Text)
    $parent = Split-Path -Parent $Path
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    [System.IO.File]::WriteAllText($Path, $Text, $script:Utf8NoBom)
}

Initialize-UnicodeConsole