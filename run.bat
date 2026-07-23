@echo off
title Cued Recall Middleware
cd /d "%~dp0"

REM ──────────────────────────────────────────────
REM  Cued Recall — one-click launcher
REM  Pass any --option to run.py, or run with no
REM  args for interactive prompts.
REM ──────────────────────────────────────────────

setlocal enabledelayedexpansion

echo === Cued Recall Memory Middleware ===
echo.

REM Check Python >= 3.11
python --version 2>nul | findstr /R "3\.1[1-9]\|3\.[2-9][0-9]" >nul
if errorlevel 1 (
    echo Python 3.11+ is required. Install it from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if --help or any -- flags are passed
set "HAS_FLAGS=0"
for %%A in (%*) do (
    echo %%A | findstr /R "^--" >nul && set HAS_FLAGS=1
)

if "%HAS_FLAGS%"=="0" if "%*"=="" (
    REM Interactive mode — ask the user
    echo No options provided. Let's set things up interactively.
    echo.

    set "MODELS_CACHE="
    set /p "MODELS_CACHE=Path to existing models directory [leave empty to download]: "
    if not "!MODELS_CACHE!"=="" (
        if not exist "!MODELS_CACHE!" (
            echo Directory not found: !MODELS_CACHE!
            set "MODELS_CACHE="
        )
    )

    set "STORAGE="
    set /p "STORAGE=Storage path (default: ./data on Windows) [Enter for default]: "

    set "REASONING="
    set /p "REASONING=Path to reasoning GGUF model [Enter for auto-download]: "

    set "JUDGE="
    set /p "JUDGE=Path to judge GGUF model [Enter for auto-download]: "

    set "EMBED="
    set /p "EMBED=Path to embedding GGUF model [Enter for auto-download]: "

    echo.
    echo Proceeding with the selected options...
    echo.

    set "ARGS="
    if not "!MODELS_CACHE!"=="" set "ARGS=!ARGS! --models-cache !MODELS_CACHE!"
    if not "!STORAGE!"==""      set "ARGS=!ARGS! --storage !STORAGE!"
    if not "!REASONING!"==""    set "ARGS=!ARGS! --reasoning-model !REASONING!"
    if not "!JUDGE!"==""        set "ARGS=!ARGS! --judge-model !JUDGE!"
    if not "!EMBED!"==""        set "ARGS=!ARGS! --embed-model !EMBED!"

    goto :run
)

REM Non-interactive: pass all args through
set "ARGS=%*"

:run

REM Install/upgrade Python dependencies
echo Installing Python dependencies...
python -m pip install --upgrade pip -q
python -m pip install -r "%~dp0cued_recall\requirements.txt" -q
if errorlevel 1 (
    echo Dependency installation failed.
    pause
    exit /b 1
)

echo.

REM Run the orchestrator
python "%~dp0run.py" %ARGS%

set EXIT_CODE=%errorlevel%

echo.
if %EXIT_CODE% equ 0 (
    echo All processes stopped cleanly.
) else (
    echo Exited with code %EXIT_CODE%. Check output above for details.
)

echo.
echo Press any key to close...
pause >nul
endlocal
