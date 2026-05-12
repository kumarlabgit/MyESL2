# setup.ps1 — Download the latest MyESL2 release binary for Windows x64
#             and place it (plus its runtime DLLs) into .\bin\.
#
# Usage (from a PowerShell prompt in the repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Tag v0.1.2
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Channel dev
#
# Channel selection:
#   - Auto-detected from `git rev-parse --abbrev-ref HEAD`: `dev` branch
#     → dev channel, anything else → main channel.
#   - Override with -Channel main|dev.
#   - main channel pulls `v*` releases via /releases/latest.
#   - dev  channel pulls `dev-v*` releases via /releases?per_page=100.
#
# On Linux/macOS use scripts/setup.sh instead.

[CmdletBinding()]
param(
    [string] $Tag     = "",
    [string] $Repo    = "kumarlabgit/MyESL2",
    [string] $Channel = ""
)

$ErrorActionPreference = "Stop"

# ── Locate repo root ─────────────────────────────────────────────
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot  = Resolve-Path (Join-Path $scriptDir "..")
$binDir    = Join-Path $repoRoot "bin"

# ── Channel autodetect ───────────────────────────────────────────
# If -Channel wasn't supplied, infer from the checked-out git branch.
# Defaults to `main` for detached HEAD, feature branches, and non-git
# checkouts so the dev channel is strictly opt-in.
if (-not $Channel) {
    $gitBranch = ""
    try {
        $gitBranch = (& git -C $repoRoot rev-parse --abbrev-ref HEAD 2>$null)
    } catch {}
    if ($LASTEXITCODE -eq 0 -and $gitBranch -eq "dev") { $Channel = "dev" } else { $Channel = "main" }
}
if ($Channel -ne "main" -and $Channel -ne "dev") {
    Write-Error "-Channel must be 'main' or 'dev' (got '$Channel')"
    exit 2
}

Write-Host "=== MyESL2 binary setup ==="
Write-Host "  Repo:     $Repo"
Write-Host "  Channel:  $Channel"
Write-Host "  Platform: Windows x64"
Write-Host "  Target:   $binDir\"
Write-Host ""

# ── Resolve release tag ──────────────────────────────────────────
# main channel: /releases/latest excludes prereleases, so dev tags
# (published with prerelease=true) can never be returned here.
# dev channel: paginate /releases and take the first `dev-v*` tag
# (the API sorts by created_at desc).
if (-not $Tag) {
    if ($Channel -eq "main") {
        Write-Host "Resolving latest main release tag..."
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
    } else {
        Write-Host "Resolving latest dev-v* release tag..."
        $apiUrl = "https://api.github.com/repos/$Repo/releases?per_page=100"
        try {
            $releases = Invoke-RestMethod -Uri $apiUrl -UseBasicParsing -Headers @{
                "User-Agent" = "myesl2-setup"
            }
        } catch {
            Write-Error "Could not query GitHub API: $apiUrl`n$_"
            exit 1
        }
        $devRelease = $releases | Where-Object { $_.tag_name -like "dev-v*" } | Select-Object -First 1
        if (-not $devRelease) { Write-Error "No dev-v* release found at $apiUrl"; exit 1 }
        $Tag = $devRelease.tag_name
    }
}

# ── Cross-contamination safety check ─────────────────────────────
# Fires on both auto-resolved tags and user-supplied -Tag values.
if ($Channel -eq "main" -and $Tag -like "dev-*") {
    Write-Error "channel=main but tag '$Tag' is a dev tag.`nPass -Channel dev to switch channels, or use a v* tag."
    exit 1
}
if ($Channel -eq "dev" -and $Tag -notlike "dev-*") {
    Write-Error "channel=dev but tag '$Tag' is not a dev tag.`nPass -Channel main to switch channels, or use a dev-v* tag."
    exit 1
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
