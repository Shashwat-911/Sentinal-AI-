@echo off
title SENTINEL AI
echo Starting SENTINEL AI...
if exist "%~dp0venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" "%~dp0app.py"
) else if exist "%~dp0venv312\Scripts\python.exe" (
    "%~dp0venv312\Scripts\python.exe" "%~dp0app.py"
) else (
    python "%~dp0app.py"
)
pause


