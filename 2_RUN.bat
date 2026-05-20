@echo off
setlocal enabledelayedexpansion
title Smart Lock - Running

echo.
echo  ============================================
echo            SMART LOCK  -  STARTING         
echo  ============================================
echo.

:: -- Guard: make sure install was run first -----------------------------------
if not exist "smartlock_env\Scripts\activate.bat" (
    echo  ERROR: Setup not found.
    echo.
    echo  Please double-click  1_INSTALL.bat  first,
    echo  then come back and run this file.
    echo.
    pause
    exit /b 1
)

:: -- Guard: make sure the script is in the same folder -----------------------
if not exist "smart-lock-code.py" (
    echo  ERROR: smart-lock-code.py not found in this folder.
    echo  Make sure all files are in the same directory.
    echo.
    pause
    exit /b 1
)

:: -- Activate environment and launch -----------------------------------------
call smartlock_env\Scripts\activate.bat
echo  Environment ready. Launching Smart Lock...
echo.

python smart-lock-code.py

:: -- On exit -----------------------------------------------------------------
echo.
echo  Smart Lock has stopped.
pause