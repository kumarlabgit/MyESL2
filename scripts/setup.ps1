# setup.ps1 — Download the latest MyESL2 release binary for Windows x64
#             and place it (plus its runtime DLLs) into .\bin\.
#
# Usage (from a PowerShell prompt in the repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Tag v0.1.2
#
# On Linux/macOS use scripts/setup.sh instead.

[CmdletBinding()]
param(
    [string] $Tag  = "",
    [string] $Repo = "kumarlabgit/MyESL2"
)

$ErrorActionPreference = "Stop"

# ── Locate repo root ─────────────────────────────────────────────
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot  = Resolve-Path (Join-Path $scriptDir "..")
$binDir    = Join-Path $repoRoot "bin"

Write-Host "=== MyESL2 binary setup ==="
Write-Host "  Repo:     $Repo"
Write-Host "  Platform: Windows x64"
Write-Host "  Target:   $binDir\"
Write-Host ""

# ── Resolve release tag ──────────────────────────────────────────
if (-not $Tag) {
    Write-Host "Resolving latest release tag..."
    $apiUrl = "https://api.github.com/repos/$Repo/releases/latest"
    try {
        $release = Invoke-RestMethod -Uri $apiUrl -UseBasicParsing -Headers @{
            "User-Agent" = "myesl2-setup"
        }
    } catch {
        Write-Error "Could not query GitHub API: $apiUrl`n$_"
        exit 1
    }
    $Tag = $release.tag_name
    if (-not $Tag) { Write-Error "No tag_name in latest release response."; exit 1 }
}
Write-Host "  Tag:      $Tag"
Write-Host ""

$baseUrl = "https://github.com/$Repo/releases/download/$Tag"

# Required assets (fail if missing)
$requiredAssets = @(
    @{ Asset = "myesl2-windows-x64.exe"; Local = "myesl2.exe"     },
    @{ Asset = "openblas.dll";           Local = "libopenblas.dll" },
    @{ Asset = "lapack.dll";             Local = "liblapack.dll"   }
)

# Fortran runtime DLLs (required by lapack.dll; older releases may not have them)
$optionalAssets = @(
    @{ Asset = "libgfortran-5.dll";      Local = "libgfortran-5.dll"  },
    @{ Asset = "libgcc_s_seh-1.dll";     Local = "libgcc_s_seh-1.dll" }
)

# ── Download into bin\ ───────────────────────────────────────────
New-Item -ItemType Directory -Force -Path $binDir | Out-Null

foreach ($a in $requiredAssets) {
    $url  = "$baseUrl/$($a.Asset)"
    $dest = Join-Path $binDir $a.Local
    Write-Host ("  v {0}  ->  bin\{1}" -f $a.Asset, $a.Local)
    try {
        Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing `
            -Headers @{ "User-Agent" = "myesl2-setup" }
    } catch {
        Write-Error "Download failed: $url`n$_"
        if (Test-Path $dest) { Remove-Item $dest -Force }
        exit 1
    }
}

foreach ($a in $optionalAssets) {
    $url  = "$baseUrl/$($a.Asset)"
    $dest = Join-Path $binDir $a.Local
    Write-Host ("  v {0}  ->  bin\{1}" -f $a.Asset, $a.Local)
    try {
        Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing `
            -Headers @{ "User-Agent" = "myesl2-setup" }
    } catch {
        Write-Host "    (not found in release - skipping)"
    }
}

# Copy data_defs.ini next to the binary so `.\bin\myesl2.exe ...` works
# from any working directory (myesl2.exe checks its own directory first).
$dataDefsSrc = Join-Path $repoRoot "data_defs.ini"
if (Test-Path $dataDefsSrc) {
    Copy-Item -Force $dataDefsSrc (Join-Path $binDir "data_defs.ini")
    Write-Host "  = data_defs.ini  ->  bin\data_defs.ini"
} else {
    Write-Warning "$dataDefsSrc not found; myesl2 will fall back to cwd lookup."
}

Write-Host ""
Write-Host "Done. Installed to $binDir\:"
Get-ChildItem $binDir | Format-Table Name, Length -AutoSize
Write-Host ""
Write-Host "Try it:"
Write-Host "  .\bin\myesl2.exe"
