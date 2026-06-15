@echo off
REM DNF Vision public deploy: backend (:8000) + frontend (:3000) + Cloudflare Tunnel.
REM The frontend proxies /api and /static to the backend, so one tunnel exposes the whole app.
cd /d "%~dp0"

echo [1/4] Starting FastAPI backend (:8000) ...
start "DNF Vision Backend" "%~dp0web\backend\run-api.bat"

echo [2/4] Starting Next.js frontend (:3000) ...
start "DNF Vision Frontend" "%~dp0web\frontend\run-dev.bat"

echo [3/4] Waiting 12s for servers to come up ...
timeout /t 12 >nul

echo [4/4] Starting Cloudflare Tunnel ...
start "DNF Vision Tunnel" cmd /k "cloudflared tunnel --url http://localhost:3000"

echo.
echo The "DNF Vision Tunnel" window prints a public URL like:
echo     https://something-random.trycloudflare.com
echo Share that URL. It stays online while these three windows are open.
echo (The quick-tunnel URL changes every restart. See web/README.md for a stable named tunnel.)
