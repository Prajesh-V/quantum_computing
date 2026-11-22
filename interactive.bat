@echo off
setlocal

rem Check Python
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not found in PATH.
    echo Please install Python 3.11+ and ensure python.exe is in your PATH.
    pause
    exit /b 1
)

python simulate.py
if errorlevel 1 (
    echo.
    echo [ERROR] Simulation exited with error.
    pause
    exit /b 1
)
exit /b 0
