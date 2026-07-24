@echo off
setlocal EnableDelayedExpansion
title Cued Recall - RAM Disk Setup
cd /d "%~dp0"

echo === Cued Recall - Windows RAM Disk Setup ===
echo.
echo This script creates a RAM disk using ImDisk for ultra-fast storage
echo of models and the block store. Data is lost on reboot.
echo.
echo Prerequisite: ImDisk Toolkit (free, open-source)
echo Download from: https://sourceforge.net/projects/imdisk-toolkit/
echo.
echo After downloading, run the installer (requires admin).
echo No reboot needed - the driver loads on install.
echo.

:check_imdisk
REM -- Locate imdisk.exe --
set "IMDISK="

REM 1) Check PATH
where imdisk 2>nul >nul
if not errorlevel 1 (
    for /f "delims=" %%I in ('where imdisk 2^>nul') do (
        if not defined IMDISK set "IMDISK=%%I"
    )
)

REM 2) Check known install locations
if not defined IMDISK (
    if exist "C:\Windows\System32\imdisk.exe" set "IMDISK=C:\Windows\System32\imdisk.exe"
)
if not defined IMDISK (
    if exist "C:\Program Files\ImDisk\imdisk.exe" set "IMDISK=C:\Program Files\ImDisk\imdisk.exe"
)
if not defined IMDISK (
    if exist "C:\Program Files (x86)\ImDisk\imdisk.exe" set "IMDISK=C:\Program Files (x86)\ImDisk\imdisk.exe"
)

REM 3) Check registry for ImDisk install path
if not defined IMDISK (
    for /f "tokens=2*" %%A in ('reg query "HKLM\SOFTWARE\ImDisk" /v InstallDir 2^>nul') do (
        if exist "%%B\imdisk.exe" set "IMDISK=%%B\imdisk.exe"
    )
)

if not defined IMDISK (
    echo [WARN] ImDisk not found.
    echo.
    echo Install it first:
    echo   1. Download: https://sourceforge.net/projects/imdisk-toolkit/
    echo   2. Run the installer ^(admin rights required^)
    echo   3. Close and reopen this terminal
    echo.
    set /p "RETRY=Press Enter after installing ImDisk, or type 'skip' to continue without: "
    if /i "!RETRY!"=="skip" goto :no_ramdisk
    goto :check_imdisk
)

echo Found: !IMDISK!
echo.

REM -- Ask for drive letter --
:ask_drive
set "DRIVE=R"
set /p "DRIVE=Drive letter for RAM disk [R]: "
if /i "!DRIVE!"=="" set "DRIVE=R"
set "DRIVE=!DRIVE:~0,1!"
if /i "!DRIVE!"=="C" echo Cannot use C:. Pick another. && goto :ask_drive
if /i "!DRIVE!"=="D" echo Cannot use D:. Pick another. && goto :ask_drive

REM Check if drive is free
fsutil fsinfo drives 2>nul | findstr /I "!DRIVE!:" >nul 2>nul
if not errorlevel 1 (
    echo Drive !DRIVE!: is already in use.
    goto :ask_drive
)

REM -- Ask for size --
:ask_size
set "SIZE=64"
set /p "SIZE=RAM disk size in GB [64]: "
if /i "!SIZE!"=="" set "SIZE=64"
echo !SIZE!| findstr /R "^[0-9][0-9]*$" >nul
if errorlevel 1 (
    echo Enter a number.
    goto :ask_size
)

REM -- Ask for NTFS block size --
echo.
echo NTFS allocation unit size (cluster size):
echo   4096  = default, good for mixed files (recommended)
echo   65536 = 64K, better for large model files ^(GGUF^)
echo   1024  = 1K, better for many small files
:ask_blocksize
set "BLOCK=4096"
set /p "BLOCK=Block size in bytes [4096]: "
if /i "!BLOCK!"=="" set "BLOCK=4096"
echo !BLOCK!| findstr /R "^[0-9][0-9]*$" >nul
if errorlevel 1 (
    echo Enter a number.
    goto :ask_blocksize
)

REM -- Ask for label --
set "LABEL=CUED_RECALL"
set /p "LABEL=Volume label [CUED_RECALL]: "
if /i "!LABEL!"=="" set "LABEL=CUED_RECALL"

echo.
echo === Summary ===
echo  Drive letter: !DRIVE!:
echo  Size:         !SIZE! GB
echo  Block size:   !BLOCK! bytes
echo  Label:        !LABEL!
echo.
echo WARNING: All existing data on this RAM disk will be lost on reboot.
echo.
set "CONFIRM="
set /p "CONFIRM=Create RAM disk? (y/n) [y]: "
if /i "!CONFIRM!"=="" set "CONFIRM=y"
if /i not "!CONFIRM!"=="y" echo Cancelled. && goto :eof

echo.
echo Creating !SIZE!G RAM disk at !DRIVE!: ...

REM Map numeric byte sizes to format.com /A: tokens (format wants 16K/32K/64K, not bytes)
set "BLOCKARG=!BLOCK!"
if "!BLOCK!"=="16384"  set "BLOCKARG=16K"
if "!BLOCK!"=="32768"  set "BLOCKARG=32K"
if "!BLOCK!"=="65536"  set "BLOCKARG=64K"
if "!BLOCK!"=="131072" set "BLOCKARG=128K"
if "!BLOCK!"=="262144" set "BLOCKARG=256K"

REM Step 1: create the raw device (no -p; imdisk's exit code does not cover format results)
"!IMDISK!" -a -s !SIZE!G -m !DRIVE!:
if errorlevel 1 (
    echo [ERROR] Failed to create RAM disk device. Run this script as Administrator.
    pause
    exit /b 1
)

REM Step 2: format explicitly so failures are visible and checkable
echo Formatting !DRIVE!: as NTFS, cluster !BLOCKARG! ...
format !DRIVE!: /FS:NTFS /Q /Y /A:!BLOCKARG! /V:!LABEL!
if errorlevel 1 (
    echo [WARN] Format with cluster size !BLOCKARG! failed, retrying with default cluster...
    format !DRIVE!: /FS:NTFS /Q /Y /V:!LABEL!
    if errorlevel 1 (
        echo [ERROR] Format failed. Removing RAM disk device.
        "!IMDISK!" -D -m !DRIVE!:
        pause
        exit /b 1
    )
)

REM Step 3: verify the volume is actually usable
if not exist "!DRIVE!:\" (
    echo [ERROR] Volume !DRIVE!: is not accessible after format.
    pause
    exit /b 1
)

echo.
echo SUCCESS: RAM disk created at !DRIVE!:  (!SIZE! GB)
echo.

REM -- Create directory structure --
echo Creating directory structure for Cued Recall...
mkdir "!DRIVE!:\cued_recall\models"
mkdir "!DRIVE!:\cued_recall\store"
if not exist "!DRIVE!:\cued_recall\store\" (
    echo [ERROR] Could not create directories on !DRIVE!: - volume may be RAW.
    pause
    exit /b 1
)
echo Directories created.

echo.
echo === Next steps ===
echo   Run Cued Recall with:
echo     run.bat --storage !DRIVE!:\cued_recall
echo.
echo   Or just: run.bat  (it will ask interactively)
echo.
echo === To remove the RAM disk ===
echo   "!IMDISK!" -d -m !DRIVE!:
echo   (or use the ImDisk GUI from Start Menu)
echo.

goto :eof

:no_ramdisk
echo Skipping RAM disk setup. Cued Recall will use local ./data directory.
echo Run: run.bat