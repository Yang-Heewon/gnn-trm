@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%download_data.ps1" %*
exit /b %ERRORLEVEL%
