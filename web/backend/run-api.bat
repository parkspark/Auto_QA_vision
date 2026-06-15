@echo off
REM FastAPI backend (:8000). cwd = web\backend so `from core import ...` works.
cd /d "%~dp0"
"%~dp0..\..\.venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000
