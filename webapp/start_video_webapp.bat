@echo off
cd /d %~dp0\..
.venv\Scripts\python.exe webapp\video_app.py
pause
