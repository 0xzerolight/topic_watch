@echo off
REM Topic Watch startup script for Windows
REM Place a shortcut to this file in shell:startup to auto-start on login
cd /d "%~dp0"
docker compose up -d
