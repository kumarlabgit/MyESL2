@echo off
REM setup.bat - Convenience wrapper that runs scripts\setup.ps1 via PowerShell.
REM Usage: scripts\setup.bat            (latest release)
REM        scripts\setup.bat -Tag v0.1.2

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
