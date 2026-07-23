@echo off
setlocal EnableDelayedExpansion
title Cued Recall — RAM Disk Setup
cd /d "%~dp0"

echo === Cued Recall — Windows RAM Disk Setup ===
echo.
echo This script creates a RAM disk using ImDisk for ultra-fast storage
echo of models and the block store. Data is lost on reboot.
echo.
echo Prerequisite: ImDisk Toolkit (free, open-source)
echo Download from: https://sourceforge.net/projects/imdisk-toolkit/
echo.
echo After downloading, run the installer (requires admin).
echo No reboot needed — the driver loads on install.
echo.

:check_imdisk
where imdisk 2>nul >nul
if errorlevel 1 (
    echo [WARN] ImDisk not found in PATH.
    echo.
    echo Install it first:
    echo   1. Download: https://sourceforge.net/projects/imdisk-toolkit/
    echo   2. Run the installer (admin rights required)
    echo   3. Close and reopen this terminal
    echo.
    set /p "RETRY=Press Enter after installing ImDisk, or type 'skip' to continue without: "
    if /i "!RETRY!"=="skip" goto :no_ramdisk
    goto :check_imdisk
)

echo ImDisk found.
echo.

REM ── Ask for drive letter ──
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

REM ── Ask for size ──
:ask_size
set "SIZE=64"
set /p "SIZE=RAM disk size in GB [64]: "
if /i "!SIZE!"=="" set "SIZE=64"
echo !SIZE!| findstr /R "^[0-9][0-9]*$" >nul
if errorlevel 1 (
    echo Enter a number.
    goto :ask_size
)

REM ── Ask for NTFS block size ──
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

REM ── Ask for label ──
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

REM Convert GB to bytes for imdisk (-s expects bytes with K/M/G suffix)
set "SIZE_ARG=!SIZE!G"

imdisk -a -s !SIZE_ARG! -m !DRIVE!: -p "/fs:ntfs /q /y /a:!BLOCK! /v:!LABEL!"
if errorlevel 1 (
    echo [ERROR] Failed to create RAM disk. Try running as Administrator.
    pause
    exit /b 1
)

echo.
echo SUCCESS: RAM disk created at !DRIVE!:  (!SIZE! GB)
echo.

REM ── Create directory structure ──
echo Creating directory structure for Cued Recall...
mkdir "!DRIVE!:\cued_recall\models" 2>nul
mkdir "!DRIVE!:\cued_recall\store" 2>nul
echo Directories created.

echo.
echo === Next steps ===
echo   Run Cued Recall with:
echo     run.bat --storage !DRIVE!:\cued_recall
echo.
echo   Or just: run.bat  (it will ask interactively)
echo.
echo === To remove the RAM disk ===
echo   imdisk -d -m !DRIVE!:
echo   (or use the ImDisk GUI from Start Menu)
echo.

goto :eof

:no_ramdisk
echo Skipping RAM disk setup. Cued Recall will use local ./data directory.
echo Run: run.bat
