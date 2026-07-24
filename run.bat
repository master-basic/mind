@echo off
title Cued Recall Middleware
cd /d "%~dp0"

REM ──────────────────────────────────────────────
REM  Cued Recall — one-click launcher
REM  Pass any --option to run.py, or run with no
REM  args for interactive prompts.
REM ──────────────────────────────────────────────

setlocal enabledelayedexpansion

REM Remembered answers from the last interactive run live here
set "SETTINGS_FILE=%~dp0run_settings.txt"

echo === Cued Recall Memory Middleware ===
echo.

REM Check Python >= 3.11
python --version 2>nul | findstr /C:"Python 3." >nul
if errorlevel 1 (
    echo Python 3.11+ is required. Install it from https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%V in ('python --version 2^>nul') do set PYVER=%%V
for /f "tokens=1,2 delims=." %%A in ("%PYVER%") do set PYMINOR=%%B
if %PYMINOR% LSS 11 (
    echo Python 3.11+ is required. You have Python %PYVER%.
    pause
    exit /b 1
)

REM Check if --help or any -- flags are passed
set "HAS_FLAGS=0"
for %%A in (%*) do (
    echo %%A | findstr /R "^--" >nul && set HAS_FLAGS=1
)

if "%HAS_FLAGS%"=="0" if "%~1"=="" goto :interactive

REM Non-interactive: pass all args through
set "ARGS=%*"
goto :run


:interactive
echo No options provided. Let's set things up interactively.
echo.

REM Load answers remembered from a previous run, if any
set "SAVED_LLAMA_BIN="
set "SAVED_MODELS_CACHE="
set "SAVED_STORAGE="
set "SAVED_REASONING="
set "SAVED_JUDGE="
set "SAVED_EMBED="
set "HAVE_SAVED=0"
if exist "%SETTINGS_FILE%" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%SETTINGS_FILE%") do set "SAVED_%%A=%%B"
    set "HAVE_SAVED=1"
)

if "!HAVE_SAVED!"=="1" goto :offer_saved
goto :ask

:offer_saved
echo Found settings from your last run:
echo   llama-server:    !SAVED_LLAMA_BIN!
echo   Models cache:    !SAVED_MODELS_CACHE!
echo   Storage path:    !SAVED_STORAGE!
echo   Reasoning model: !SAVED_REASONING!
echo   Judge model:     !SAVED_JUDGE!
echo   Embedding model: !SAVED_EMBED!
echo.
set "REUSE=Y"
set /p "REUSE=Reuse these settings? (Y/n) [Y]: "
if /i "!REUSE!"=="" set "REUSE=Y"
if /i not "!REUSE!"=="Y" goto :ask
set "LLAMA_BIN=!SAVED_LLAMA_BIN!"
set "MODELS_CACHE=!SAVED_MODELS_CACHE!"
set "STORAGE=!SAVED_STORAGE!"
set "REASONING=!SAVED_REASONING!"
set "JUDGE=!SAVED_JUDGE!"
set "EMBED=!SAVED_EMBED!"
goto :save_and_build

:ask
REM Each prompt offers the previously saved value as its default (press Enter to keep)
set "LLAMA_BIN=!SAVED_LLAMA_BIN!"
set /p "LLAMA_BIN=Folder containing llama-server.exe [!SAVED_LLAMA_BIN!] (blank=auto-detect from PATH): "
if not "!LLAMA_BIN!"=="" if not exist "!LLAMA_BIN!" echo [WARN] Not found: !LLAMA_BIN! - will pass it anyway

set "MODELS_CACHE=!SAVED_MODELS_CACHE!"
set /p "MODELS_CACHE=Path to existing models directory [!SAVED_MODELS_CACHE!]: "
if not "!MODELS_CACHE!"=="" if not exist "!MODELS_CACHE!" (
    echo Directory not found: !MODELS_CACHE!
    set "MODELS_CACHE="
)

set "STORAGE=!SAVED_STORAGE!"
set /p "STORAGE=Storage path (default: ./data on Windows) [!SAVED_STORAGE!]: "

set "REASONING=!SAVED_REASONING!"
set /p "REASONING=Path to reasoning GGUF model [!SAVED_REASONING!]: "

set "JUDGE=!SAVED_JUDGE!"
set /p "JUDGE=Path to judge GGUF model [!SAVED_JUDGE!]: "

set "EMBED=!SAVED_EMBED!"
set /p "EMBED=Path to embedding GGUF model [!SAVED_EMBED!]: "

:save_and_build
echo.
echo Proceeding with the selected options...
echo.

REM Remember these answers for next time (redirection first avoids the
REM trailing-digit stream-number trap; separate echoes avoid a ( ) block
REM so parentheses in Windows paths can't truncate the write)
>"%SETTINGS_FILE%"  echo LLAMA_BIN=!LLAMA_BIN!
>>"%SETTINGS_FILE%" echo MODELS_CACHE=!MODELS_CACHE!
>>"%SETTINGS_FILE%" echo STORAGE=!STORAGE!
>>"%SETTINGS_FILE%" echo REASONING=!REASONING!
>>"%SETTINGS_FILE%" echo JUDGE=!JUDGE!
>>"%SETTINGS_FILE%" echo EMBED=!EMBED!

set "ARGS="
if not "!LLAMA_BIN!"=="" set "ARGS=!ARGS! --llama-bin !LLAMA_BIN!"
if not "!MODELS_CACHE!"=="" set "ARGS=!ARGS! --models-cache !MODELS_CACHE!"
if not "!STORAGE!"==""      set "ARGS=!ARGS! --storage !STORAGE!"
if not "!REASONING!"==""    set "ARGS=!ARGS! --reasoning-model !REASONING!"
if not "!JUDGE!"==""        set "ARGS=!ARGS! --judge-model !JUDGE!"
if not "!EMBED!"==""        set "ARGS=!ARGS! --embed-model !EMBED!"

goto :run

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
