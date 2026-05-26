@echo off
cd /d "%~dp0"
echo 正在运行验证脚本...
echo.
".venv\Scripts\python.exe" validate_loss_functions.py
echo.
pause
