# Topic Watch installer for Windows
# Usage: irm https://raw.githubusercontent.com/0xzerolight/topic_watch/main/scripts/install.ps1 | iex
#   or:  powershell -ExecutionPolicy Bypass -File install.ps1
#
# SUPPLY-CHAIN NOTE (OVH-146): irm|iex runs whatever this URL returns, and by
# default this script also fetches docker-compose.prod.yml (which selects the
# container image) from the same ref. Both are pulled from the mutable "main"
# branch with no commit pin, tag, signature, or checksum, so a repo/branch
# compromise or a MITM proxy means arbitrary code runs as you. To reduce trust:
#   1. Review this script before piping it to iex, or download + run it.
#   2. Pin a specific commit or release tag instead of "main":
#        $env:TOPIC_WATCH_REF="v1.1.2"; irm `
#          https://raw.githubusercontent.com/0xzerolight/topic_watch/v1.1.2/scripts/install.ps1 | iex
#      TOPIC_WATCH_REF also pins the docker-compose file this script downloads.

$ErrorActionPreference = 'Stop'

$Repo = "0xzerolight/topic_watch"
# Pin to a commit SHA or release tag for a verifiable install (OVH-146).
# Defaults to "main" (mutable) — see the supply-chain note above.
$Branch = if ($env:TOPIC_WATCH_REF) { $env:TOPIC_WATCH_REF } else { "main" }
$InstallDir = if ($env:TOPIC_WATCH_DIR) { $env:TOPIC_WATCH_DIR } else { Join-Path $env:LOCALAPPDATA "TopicWatch" }
$Port = if ($env:TOPIC_WATCH_PORT) { $env:TOPIC_WATCH_PORT } else { "8000" }
# Login autostart is opt-in (OVH-147). Set TOPIC_WATCH_AUTOSTART=yes|no to answer
# non-interactively; default in a non-interactive (piped) run is "no".
$Autostart = $env:TOPIC_WATCH_AUTOSTART

function Write-Info($msg)  { Write-Host "[+] $msg" -ForegroundColor Green }
function Write-Warn($msg)  { Write-Host "[!] $msg" -ForegroundColor Yellow }
function Write-Err($msg)   { Write-Host "[x] $msg" -ForegroundColor Red }

# --- Prerequisite checks ---
try {
    $null = & docker compose version 2>&1
    if ($LASTEXITCODE -ne 0) { throw }
} catch {
    Write-Err "Docker with Compose plugin is required but not found."
    Write-Host ""
    Write-Host "Install Docker Desktop: https://docs.docker.com/desktop/install/windows-install/"
    exit 1
}

$dockerVersion = (docker compose version 2>&1) | Select-Object -First 1
Write-Info "Docker found: $dockerVersion"

# --- Create install directory ---
Write-Info "Installing to $InstallDir"
New-Item -ItemType Directory -Path (Join-Path $InstallDir "data") -Force | Out-Null

# --- Download production compose file ---
$ComposeUrl = "https://raw.githubusercontent.com/$Repo/$Branch/docker-compose.prod.yml"
$ComposeDest = Join-Path $InstallDir "docker-compose.yml"
Write-Info "Downloading docker-compose.yml..."
Invoke-WebRequest -Uri $ComposeUrl -OutFile $ComposeDest -UseBasicParsing

# --- Pull and start ---
Push-Location $InstallDir
try {
    Write-Info "Pulling Docker image..."
    & docker compose pull
    if ($LASTEXITCODE -ne 0) { throw "docker compose pull failed" }

    Write-Info "Starting Topic Watch..."
    & docker compose up -d
    if ($LASTEXITCODE -ne 0) { throw "docker compose up failed" }
} finally {
    Pop-Location
}

# --- Wait for health check ---
Write-Info "Waiting for Topic Watch to start..."
$healthy = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$Port/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $healthy = $true
            break
        }
    } catch {}
    Start-Sleep -Seconds 1
}

if (-not $healthy) {
    Write-Warn "Health check not responding yet. Check: docker compose -f `"$ComposeDest`" logs"
}

# --- Desktop integration ---

# Start Menu shortcut (opens browser to Topic Watch)
$StartMenuDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
$ShortcutPath = Join-Path $StartMenuDir "Topic Watch.lnk"
try {
    $WshShell = New-Object -ComObject WScript.Shell
    $Shortcut = $WshShell.CreateShortcut($ShortcutPath)
    $Shortcut.TargetPath = "http://localhost:$Port"
    $Shortcut.Description = "Self-hosted news monitoring with AI-powered novelty detection"
    $Shortcut.Save()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($WshShell) | Out-Null
    Write-Info "Start Menu shortcut installed (search 'Topic Watch' in Start)"
} catch {
    Write-Warn "Could not create Start Menu shortcut: $_"
}

# --- Autostart on login (opt-in, OVH-147) ---
# A Startup-folder shortcut runs Topic Watch on every login. That is real
# persistence, so ask first instead of installing it silently. Non-interactive
# runs default to "no".
$wantAutostart = $false
switch -Regex ($Autostart) {
    '^(?i)(yes|y)$' { $wantAutostart = $true }
    '^(?i)(no|n)$'  { $wantAutostart = $false }
    default {
        if ([Environment]::UserInteractive -and -not [Console]::IsInputRedirected) {
            $reply = Read-Host "Start Topic Watch automatically on login (Startup-folder shortcut)? [y/N]"
            if ($reply -match '^(?i)(yes|y)$') { $wantAutostart = $true }
        } else {
            Write-Warn "Skipping login autostart (non-interactive). Set TOPIC_WATCH_AUTOSTART=yes to enable it."
        }
    }
}

$StartupDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup"
$StartupShortcut = Join-Path $StartupDir "Topic Watch.lnk"
if ($wantAutostart) {
    # Startup batch script (auto-start on login)
    $StartupBat = Join-Path $InstallDir "start-topic-watch.bat"
    @"
@echo off
cd /d "$InstallDir"
docker compose up -d
"@ | Set-Content -Path $StartupBat -Encoding ASCII

    # Shortcut in Startup folder pointing to the batch script
    try {
        $WshShell = New-Object -ComObject WScript.Shell
        $Shortcut = $WshShell.CreateShortcut($StartupShortcut)
        $Shortcut.TargetPath = $StartupBat
        $Shortcut.WorkingDirectory = $InstallDir
        $Shortcut.WindowStyle = 7  # Minimized
        $Shortcut.Description = "Start Topic Watch on login"
        $Shortcut.Save()
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($WshShell) | Out-Null
        Write-Info "Startup shortcut installed (Topic Watch will start on login)"
        Write-Info "To remove autostart later: delete `"$StartupShortcut`""
    } catch {
        Write-Warn "Could not create startup shortcut: $_"
    }
} else {
    Write-Info "Login autostart not installed. Enable later by re-running with TOPIC_WATCH_AUTOSTART=yes."
}

# --- Open browser ---
Write-Host ""
Write-Info "Topic Watch is running!"
Write-Host ""
Write-Host "  Open http://localhost:$Port to complete setup."
Write-Host "  Data stored in: $(Join-Path $InstallDir 'data')"
Write-Host ""
Write-Host "  Manage with:"
Write-Host "    cd `"$InstallDir`"; docker compose logs      # View logs"
Write-Host "    cd `"$InstallDir`"; docker compose restart   # Restart"
Write-Host "    cd `"$InstallDir`"; docker compose down      # Stop"
Write-Host ""
Write-Host "  Uninstall:"
Write-Host "    cd `"$InstallDir`"; docker compose down       # Stop the container"
Write-Host "    Remove-Item `"$StartupShortcut`"              # Remove login autostart (if enabled)"
Write-Host "    Remove-Item `"$ShortcutPath`"                 # Remove Start Menu shortcut"
Write-Host "    Remove-Item -Recurse -Force `"$InstallDir`"   # Remove install dir + data (irreversible)"
Write-Host ""

Start-Process "http://localhost:$Port"
