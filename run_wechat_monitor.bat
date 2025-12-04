@echo off
REM Run WeChat monitor from the repo folder with virtualenv

cd /d "%~dp0"

IF NOT EXIST ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    echo Please create it first with:
    echo    python -m venv .venv
    echo    .venv\Scripts\activate
    echo    (then install whatever deps you need)
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo [INFO] Starting WeChat monitor...
python wechat_monitor.py

echo.
echo [INFO] WeChat monitor exited. Press any key to close this window.
pause >nul
