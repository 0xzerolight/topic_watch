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
# Host interface the container's port is published on. Loopback by default:
# Topic Watch has no authentication, so binding every interface would expose an
# unauthenticated app to the whole network. Asked interactively below.
$BindAddr = $env:TOPIC_WATCH_BIND_ADDR
# Login autostart is opt-in (OVH-147). Set TOPIC_WATCH_AUTOSTART=yes|no to answer
# non-interactively; default in a non-interactive (piped) run is "no".
# Normalized to "" rather than $null so the switch below reliably reaches its
# default (prompt) branch when the variable is unset.
$Autostart = if ($env:TOPIC_WATCH_AUTOSTART) { $env:TOPIC_WATCH_AUTOSTART } else { "" }

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

# --- Setup questions ---
# Asked up front so the install runs uninterrupted afterwards. Every question is
# skipped when its environment variable is already set, which keeps automated and
# re-run installs reproducible. With no console, each falls back to its default.
$Interactive = [Environment]::UserInteractive -and -not [Console]::IsInputRedirected

# 1. Network exposure. Loopback unless the user opts into LAN access.
if (-not $BindAddr) {
    if ($Interactive) {
        Write-Host ""
        Write-Host "Who should be able to reach Topic Watch?"
        Write-Host "  1) This computer only          (recommended)"
        Write-Host "  2) Any device on my network    (needs a reverse proxy to be safe)"
        $choice = Read-Host "Choice [1]"
        $BindAddr = if ($choice -eq "2") { "0.0.0.0" } else { "127.0.0.1" }
    } else {
        $BindAddr = "127.0.0.1"
    }
}

if ($BindAddr -eq "0.0.0.0") {
    Write-Warn "Topic Watch will be reachable from your whole network."
    Write-Warn "It has no login screen: anyone who can reach this machine can read your"
    Write-Warn "topics and spend your LLM API budget. Beyond a trusted home network, put"
    Write-Warn "it behind a reverse proxy with authentication - see SECURITY.md."
}

# 2. Autostart on login. Persistence stays opt-in for unattended runs (OVH-147),
#    but an interactive user is asked directly and the recommended answer is yes:
#    Topic Watch checks topics on a schedule, so without it monitoring stops after
#    a reboot and nothing says so.
$wantAutostart = $false
switch -Regex ($Autostart) {
    '^(?i)(yes|y)$' { $wantAutostart = $true }
    '^(?i)(no|n)$'  { $wantAutostart = $false }
    default {
        if ($Interactive) {
            Write-Host ""
            Write-Host "Start Topic Watch automatically on login?"
            Write-Host "  Recommended: it checks topics on a schedule, so without this it"
            Write-Host "  stops monitoring after a reboot until you start it by hand."
            $reply = Read-Host "Enable autostart? [Y/n]"
            $wantAutostart = -not ($reply -match '^(?i)(n|no)$')
        } else {
            Write-Warn "Skipping login autostart (non-interactive). Set TOPIC_WATCH_AUTOSTART=yes to enable it."
        }
    }
}

# 3. Port, asked only when the default is already taken.
if (-not $env:TOPIC_WATCH_PORT) {
    # Fail open: this check is a convenience, and $ErrorActionPreference='Stop'
    # would otherwise let a missing Get-NetTCPConnection abort the whole install.
    $inUse = $false
    try {
        $inUse = $null -ne (Get-NetTCPConnection -LocalPort ([int]$Port) -State Listen -ErrorAction SilentlyContinue)
    } catch {
        $inUse = $false
    }
    if ($inUse) {
        if ($Interactive) {
            Write-Host ""
            Write-Warn "Port $Port is already in use on this machine."
            $chosen = Read-Host "Use a different port [8080]"
            if (-not $chosen) { $chosen = "8080" }
            if ($chosen -match '^\d+$') { $Port = $chosen }
            else { Write-Warn "Not a port number - keeping $Port; the install may fail." }
        } else {
            Write-Warn "Port $Port is already in use - the install will likely fail."
            Write-Warn "Set TOPIC_WATCH_PORT to choose another and re-run."
        }
    }
}

# --- Create install directory ---
Write-Info "Installing to $InstallDir"
New-Item -ItemType Directory -Path (Join-Path $InstallDir "data") -Force | Out-Null

# --- Download production compose file ---
$ComposeUrl = "https://raw.githubusercontent.com/$Repo/$Branch/docker-compose.prod.yml"
$ComposeDest = Join-Path $InstallDir "docker-compose.yml"
Write-Info "Downloading docker-compose.yml..."
Invoke-WebRequest -Uri $ComposeUrl -OutFile $ComposeDest -UseBasicParsing

# Also fetch the Ollama/local-LLM override example so the README's documented
# override-file step works from a script install too, not only a source
# checkout. Optional (only needed for local LLM providers), so a failure here
# warns instead of aborting the install.
try {
    $OverrideUrl = "https://raw.githubusercontent.com/$Repo/$Branch/docker-compose.override.example.yml"
    $OverrideDest = Join-Path $InstallDir "docker-compose.override.example.yml"
    Invoke-WebRequest -Uri $OverrideUrl -OutFile $OverrideDest -UseBasicParsing
} catch {
    Write-Warn "Could not download docker-compose.override.example.yml (only needed for Ollama/local LLM setups)."
}

# --- Persist the answers to .env ---
# The compose file reads these. Writing them here is what makes the answers above
# survive a later `docker compose up -d` and any future re-run of this installer:
# .env is upserted key by key, whereas docker-compose.yml is overwritten.
$EnvFile = Join-Path $InstallDir ".env"

function Set-EnvVar($key, $value, $file) {
    $line = "$key=$value"
    if (Test-Path $file) {
        $kept = @(Get-Content $file | Where-Object { $_ -notmatch "^$([regex]::Escape($key))=" })
        Set-Content -Path $file -Value ($kept + $line) -Encoding ASCII
    } else {
        Set-Content -Path $file -Value $line -Encoding ASCII
    }
}

Set-EnvVar "TOPIC_WATCH_PORT" $Port $EnvFile
Set-EnvVar "TOPIC_WATCH_BIND_ADDR" $BindAddr $EnvFile
Write-Info "Wrote port and access settings to .env"

# --- Pull and start ---
Push-Location $InstallDir
try {
    Write-Info "Pulling Docker image..."
    & docker compose pull
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Could not pull the Docker image (ghcr.io/$Repo)."
        Write-Host ""
        Write-Host "  Most likely the image is not publicly accessible, or ghcr.io is unreachable."
        Write-Host "  - Check your network and that https://ghcr.io is reachable."
        Write-Host "  - Maintainers: confirm the GHCR package visibility is set to Public."
        Write-Host "  - Pin a known release instead of latest: set TOPIC_WATCH_REF=<tag> and re-run."
        throw "docker compose pull failed"
    }

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

# AUG-059: a failed health check must not be reported as a successful
# install — stop here, before Start Menu/startup shortcuts or the "running!"
# message, so a broken install never looks like a working one.
if (-not $healthy) {
    Write-Err "Health check did not pass after starting Topic Watch."
    Write-Host "  Diagnose with: docker compose -f `"$ComposeDest`" logs"
    exit 1
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
# persistence, so it is never installed silently: $wantAutostart was decided by
# the question above, or by TOPIC_WATCH_AUTOSTART, and defaults to false when
# there is no console.
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
if ($BindAddr -eq "127.0.0.1") {
    Write-Host "  Reachable from: this computer only"
    Write-Host "    To allow other devices, set TOPIC_WATCH_BIND_ADDR=0.0.0.0 in"
    Write-Host "    $EnvFile and run: docker compose up -d"
} else {
    Write-Host "  Reachable from: any device on your network (no login required)"
}
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
