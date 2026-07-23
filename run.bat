@echo off
title Cued Recall Middleware
cd /d "%~dp0"

echo === Cued Recall Memory Middleware ===
echo.

REM Check Python >= 3.11
python --version 2>nul | findstr /R "3\.1[1-9]\|3\.[2-9][0-9]" >nul
if errorlevel 1 (
    echo Python 3.11+ is required. Install it from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Install Python dependencies
echo Installing Python dependencies...
python -m pip install --upgrade pip -q
python -m pip install -r "%~dp0cued_recall\requirements.txt" -q

REM Run the orchestrator
python "%~dp0run.py"

echo.
echo Press any key to exit...
pause >nul
