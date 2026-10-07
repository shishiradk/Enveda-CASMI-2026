@echo off
rem Verifies the exported networks and writes out\export\check_summary.txt (send this text back).
setlocal
cd /d "%~dp0"
venv\Scripts\python check_result.py --data . --out out
echo.
echo The summary above is saved in out\export\check_summary.txt
pause
