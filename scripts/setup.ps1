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

$baseUrl       = "https://github.com/$Repo/releases/download/$Tag"
$archiveAsset  = "myesl2-windows-x64.zip"
$archiveUrl    = "$baseUrl/$archiveAsset"

# ── Download archive and extract into bin\ ──────────────────────
New-Item -ItemType Directory -Force -Path $binDir | Out-Null

$tmpZip = Join-Path ([System.IO.Path]::GetTempPath()) "myesl2-$([System.Guid]::NewGuid()).zip"
Write-Host ("  v {0}" -f $archiveAsset)
try {
    Invoke-WebRequest -Uri $archiveUrl -OutFile $tmpZip -UseBasicParsing `
        -Headers @{ "User-Agent" = "myesl2-setup" }
} catch {
    Write-Error "Download failed: $archiveUrl`n$_"
    if (Test-Path $tmpZip) { Remove-Item $tmpZip -Force }
    exit 1
}

Write-Host "  -> extracting into bin\"
Expand-Archive -Path $tmpZip -DestinationPath $binDir -Force
Remove-Item $tmpZip -Force

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
