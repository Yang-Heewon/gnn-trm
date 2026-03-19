@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%download_cwq_vocab_only.ps1" %*
exit /b %ERRORLEVEL%
