@echo off
setlocal enabledelayedexpansion
title Smart Lock - One-Time Setup
echo.
echo  ===================================================
echo          SMART LOCK  -  ONE-TIME SETUP
echo      Run this ONCE, then use 2_RUN.bat
echo  ===================================================
echo.

:: -- STEP 1: Find a compatible Python ----------------------------------------
echo  [1/5]  Locating a compatible Python version...

set "PY_VER="

:: Check Windows Python Launcher versions (3.11 preferred, then 3.10, 3.9)
py -3.11 --version >nul 2>&1
if not errorlevel 1 ( set "PY_VER=3.11" & goto :found )

py -3.10 --version >nul 2>&1
if not errorlevel 1 ( set "PY_VER=3.10" & goto :found )

py -3.9 --version >nul 2>&1
if not errorlevel 1 ( set "PY_VER=3.9" & goto :found )

:: Fallback: bare "python" - check its minor version
python --version >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=2 delims= " %%V in ('python --version 2^>^&1') do set "RAWVER=%%V"
    for /f "tokens=2 delims=." %%M in ("!RAWVER!") do set "PY_MINOR=%%M"
    if "!PY_MINOR!"=="9"  ( set "PY_VER=fallback" & goto :found )
    if "!PY_MINOR!"=="10" ( set "PY_VER=fallback" & goto :found )
    if "!PY_MINOR!"=="11" ( set "PY_VER=fallback" & goto :found )
)

echo.
echo  ERROR: No compatible Python version found.
echo  -------------------------------------------------------
echo   TensorFlow requires Python 3.9, 3.10, or 3.11.
echo.
echo   Download Python 3.11 (recommended):
echo     https://www.python.org/downloads/release/python-3119/
echo.
echo   Scroll to "Files" and choose:
echo     Windows installer (64-bit)
echo.
echo   IMPORTANT: Check "Add python.exe to PATH" on the
echo   first screen of the installer.
echo  -------------------------------------------------------
goto :end

:found
if "!PY_VER!"=="fallback" (
    for /f "tokens=*" %%V in ('python --version 2^>^&1') do echo         Found: %%V
) else (
    for /f "tokens=*" %%V in ('py -!PY_VER! --version 2^>^&1') do echo         Found: %%V  [via: py -!PY_VER!]
    echo !PY_VER!> smartlock_pycmd.txt
)
echo.

:: -- STEP 2: Check pip -------------------------------------------------------
echo  [2/5]  Checking pip...
if "!PY_VER!"=="fallback" (
    python -m pip --version >nul 2>&1
) else (
    py -!PY_VER! -m pip --version >nul 2>&1
)
if errorlevel 1 (
    echo  ERROR: pip is not available. Please reinstall Python.
    goto :end
)
echo         pip OK.
echo.

:: -- STEP 3: Create virtual environment --------------------------------------
echo  [3/5]  Creating isolated environment (smartlock_env)...
if exist smartlock_env (
    echo         Environment already exists - skipping creation.
    goto :venv_done
)

if "!PY_VER!"=="fallback" (
    python -m venv smartlock_env
) else (
    py -!PY_VER! -m venv smartlock_env
)
if errorlevel 1 (
    echo  ERROR: Could not create virtual environment.
    goto :end
)
echo         Done.

:venv_done
echo.

:: -- STEP 4: Activate and upgrade pip ----------------------------------------
echo  [4/5]  Activating environment and upgrading pip...
call smartlock_env\Scripts\activate.bat
if errorlevel 1 (
    echo  ERROR: Could not activate virtual environment.
    goto :end
)
python -m pip install --upgrade pip --quiet
echo         Done.
echo.

:: -- STEP 5: Install packages ------------------------------------------------

echo  [5/5]  Installing packages (first run takes 3-5 minutes)...

echo         Please wait - do not close this window.

echo.

pip install tensorflow opencv-python keras-facenet scikit-learn matplotlib scipy dlib-bin

if errorlevel 1 (

    echo.

    echo  ERROR: Main package installation failed.

    goto :end

)

echo.

echo         Installing dlib prebuilt wheel...

pip install dlib-bin

if errorlevel 1 (

    echo.

    echo  ERROR: dlib installation failed.

    goto :end

)

echo         All packages installed.

echo.
echo  +==========================================+
echo  ^|            SETUP COMPLETE!  OK           ^|
echo  ^|                                          ^|
echo  ^|  Double-click  2_RUN.bat  to start the   ^|
echo  ^|  Smart Lock any time.                    ^|
echo  +==========================================+
echo.

:end
echo  Press any key to close...
pause >nul
