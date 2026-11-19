@echo off
setlocal
cd /d "%~dp0"
python demo_ai.py
set "seme_exit=%errorlevel%"
echo.
pause
exit /b %seme_exit%
