@echo off
REM DNF Vision web app launcher: backend (:8000) + frontend (:3000) in separate windows.
cd /d "%~dp0"

echo [1/3] Starting FastAPI backend (:8000) ...
start "DNF Vision Backend" "%~dp0web\backend\run-api.bat"

echo [2/3] Starting Next.js frontend (:3000) ...
start "DNF Vision Frontend" "%~dp0web\frontend\run-dev.bat"

echo [3/3] Opening browser in 8 seconds ...
timeout /t 8 >nul
start http://localhost:3000

echo.
echo Two console windows opened. Close each window to stop the servers.
