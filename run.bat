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

REM %DATE%/%TIME% are locale-formatted and unsortable; WMIC/PowerShell give a
REM stable YYYY-MM-DD HH:MM:SS. Fall back to the raw locale values if neither
REM is available, so a missing WMIC never stops the launcher.
set "LAUNCH_TS="
for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format \"yyyy-MM-dd_HH:mm:ss\"" 2^>nul') do set "LAUNCH_TS=%%t"
if defined LAUNCH_TS set "LAUNCH_TS=!LAUNCH_TS:_= !"
if not defined LAUNCH_TS set "LAUNCH_TS=%DATE% %TIME%"

echo === Cued Recall Memory Middleware ===
echo     launched !LAUNCH_TS!
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
set "SAVED_REASONING_CHOICE="
set "SAVED_JUDGE="
set "SAVED_EMBED="
set "SAVED_HOST="
set "HAVE_SAVED=0"
if exist "%SETTINGS_FILE%" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%SETTINGS_FILE%") do set "SAVED_%%A=%%B"
    set "HAVE_SAVED=1"
)

if "!HAVE_SAVED!"=="1" goto :offer_saved
goto :ask

:offer_saved
REM run.py records what it resolved (MODEL_<n>_NAME / MODEL_<n>_CTX per catalog
REM choice, plus JUDGE_NAME / EMBED_NAME). The choice number is only known at
REM runtime, so reaching those keys needs a second expansion pass -- that is
REM what `call set` with %%..%% does. Context is stored per model because each
REM one autosizes to a different window.
set "CUR_NAME=!SAVED_REASONING_NAME!"
set "CUR_CTX=!SAVED_REASONING_CTX!"
if not "!SAVED_REASONING_CHOICE!"=="" call set "CUR_NAME=%%SAVED_MODEL_!SAVED_REASONING_CHOICE!_NAME%%"
if not "!SAVED_REASONING_CHOICE!"=="" call set "CUR_CTX=%%SAVED_MODEL_!SAVED_REASONING_CHOICE!_CTX%%"

echo Found settings from your last run:
echo   llama-server:    !SAVED_LLAMA_BIN!
echo   Models cache:    !SAVED_MODELS_CACHE!
echo   Storage path:    !SAVED_STORAGE!
if "!SAVED_HOST!"=="" (echo   Listen on:       127.0.0.1 ^(this PC only^)) else (echo   Listen on:       !SAVED_HOST!)
echo.
echo Models it will load:
if not "!CUR_NAME!"=="" echo   Reasoning:       !CUR_NAME!   [choice !SAVED_REASONING_CHOICE!, ctx !CUR_CTX!]
if "!CUR_NAME!"=="" echo   Reasoning:       choice #!SAVED_REASONING_CHOICE! - name recorded after the next launch
if not "!SAVED_JUDGE_NAME!"=="" echo   Judge:           !SAVED_JUDGE_NAME!
if not "!SAVED_EMBED_NAME!"=="" echo   Embedding:       !SAVED_EMBED_NAME!
REM Explicit paths override the catalog, so show them when the user set any.
if not "!SAVED_REASONING!"=="" echo   Reasoning path:  !SAVED_REASONING!
if not "!SAVED_JUDGE!"=="" echo   Judge path:      !SAVED_JUDGE!
if not "!SAVED_EMBED!"=="" echo   Embedding path:  !SAVED_EMBED!
echo.
set "REUSE=Y"
set /p "REUSE=Reuse these settings? (Y/n) [Y]: "
if /i "!REUSE!"=="" set "REUSE=Y"
if /i not "!REUSE!"=="Y" goto :ask
set "LLAMA_BIN=!SAVED_LLAMA_BIN!"
set "MODELS_CACHE=!SAVED_MODELS_CACHE!"
set "STORAGE=!SAVED_STORAGE!"
set "REASONING=!SAVED_REASONING!"
set "REASONING_CHOICE=!SAVED_REASONING_CHOICE!"
set "JUDGE=!SAVED_JUDGE!"
set "EMBED=!SAVED_EMBED!"
set "HOST=!SAVED_HOST!"
goto :save_and_build

:ask
REM Each prompt offers the previously saved value as its default (press Enter to keep)
REM The model-menu choice is not prompted for here, but it still has to survive:
REM without this, answering "n" above dropped it from the rewrite below and
REM silently lost the remembered model.
set "REASONING_CHOICE=!SAVED_REASONING_CHOICE!"

set "LLAMA_BIN=!SAVED_LLAMA_BIN!"
set /p "LLAMA_BIN=Folder containing llama-server.exe [!SAVED_LLAMA_BIN!] (blank=auto-detect from PATH): "
if not "!LLAMA_BIN!"=="" if not exist "!LLAMA_BIN!" echo [WARN] Not found: !LLAMA_BIN! - will pass it anyway

set "MODELS_CACHE=!SAVED_MODELS_CACHE!"
set /p "MODELS_CACHE=Folder on your hard drive to KEEP models (downloaded here once, copied to RAM each run) [!SAVED_MODELS_CACHE!] (blank=store in storage dir): "

set "STORAGE=!SAVED_STORAGE!"
set /p "STORAGE=Storage path (default: ./data on Windows) [!SAVED_STORAGE!]: "

set "REASONING=!SAVED_REASONING!"
set /p "REASONING=Path to reasoning GGUF model [!SAVED_REASONING!]: "

set "JUDGE=!SAVED_JUDGE!"
set /p "JUDGE=Path to judge GGUF model [!SAVED_JUDGE!]: "

set "EMBED=!SAVED_EMBED!"
set /p "EMBED=Path to embedding GGUF model [!SAVED_EMBED!]: "

REM 0.0.0.0 opens the chat UI, admin page and API to every machine on the
REM network. Nothing in front of it asks for a password, so this is a LAN-only
REM answer, and Windows Firewall still has to allow the port.
echo.
echo Who can reach this? 127.0.0.1 = this PC only, 0.0.0.0 = anyone on your network (no password).
set "HOST=!SAVED_HOST!"
set /p "HOST=Listen address [!SAVED_HOST!] (blank=127.0.0.1): "

:save_and_build
echo.
echo Proceeding with the selected options...
echo.

REM The rewrite below starts the file from scratch, which would drop the keys
REM run.py records (the per-model names and context sizes). They cannot be
REM listed by name -- MODEL_4_NAME only exists once model 4 has been launched --
REM so set them aside and append them back afterwards. Matching by prefix means
REM a new catalog entry needs no change here.
set "KEEP_FILE=%TEMP%\cued_recall_keep.txt"
del "%KEEP_FILE%" 2>nul
if exist "%SETTINGS_FILE%" findstr /B "MODEL_ JUDGE_NAME EMBED_NAME REASONING_NAME REASONING_CTX" "%SETTINGS_FILE%" >"%KEEP_FILE%" 2>nul

REM Remember these answers for next time (redirection first avoids the
REM trailing-digit stream-number trap; separate echoes avoid a ( ) block
REM so parentheses in Windows paths can't truncate the write)
>"%SETTINGS_FILE%"  echo LLAMA_BIN=!LLAMA_BIN!
>>"%SETTINGS_FILE%" echo MODELS_CACHE=!MODELS_CACHE!
>>"%SETTINGS_FILE%" echo STORAGE=!STORAGE!
>>"%SETTINGS_FILE%" echo REASONING=!REASONING!
>>"%SETTINGS_FILE%" echo JUDGE=!JUDGE!
>>"%SETTINGS_FILE%" echo EMBED=!EMBED!
>>"%SETTINGS_FILE%" echo HOST=!HOST!
REM Preserve the remembered model-menu choice; run.py rewrites it when the
REM user picks from the menu.
if not "!REASONING_CHOICE!"=="" (>>"%SETTINGS_FILE%" echo REASONING_CHOICE=!REASONING_CHOICE!)
if exist "%KEEP_FILE%" type "%KEEP_FILE%" >>"%SETTINGS_FILE%"
del "%KEEP_FILE%" 2>nul

set "ARGS="
if not "!LLAMA_BIN!"=="" set "ARGS=!ARGS! --llama-bin !LLAMA_BIN!"
if not "!MODELS_CACHE!"=="" set "ARGS=!ARGS! --models-cache !MODELS_CACHE!"
if not "!STORAGE!"==""      set "ARGS=!ARGS! --storage !STORAGE!"
if not "!REASONING!"==""    set "ARGS=!ARGS! --reasoning-model !REASONING!"
if not "!REASONING_CHOICE!"=="" set "ARGS=!ARGS! --reasoning-choice !REASONING_CHOICE!"
if not "!JUDGE!"==""        set "ARGS=!ARGS! --judge-model !JUDGE!"
if not "!EMBED!"==""        set "ARGS=!ARGS! --embed-model !EMBED!"
if not "!HOST!"==""         set "ARGS=!ARGS! --host !HOST!"

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
