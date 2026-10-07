@echo off
rem Starts the training, or continues it from the last checkpoint if it was interrupted.
rem Just run this file again after any interruption. Optional: run_train.bat --bs 128
setlocal
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
  echo PROBLEM: run setup_windows.bat first.
  pause
  exit /b 1
)
venv\Scripts\python train_fp.py --data . --out out %*
set RC=%ERRORLEVEL%
echo.
if %RC%==0 echo FINISHED. Now run check_result.bat
if %RC%==2 echo NOT FINISHED YET. Run run_train.bat again to continue.
if %RC%==3 echo OUT OF GPU MEMORY. Run:  run_train.bat --bs 128
if not %RC%==0 if not %RC%==2 if not %RC%==3 echo STOPPED WITH AN ERROR. Run run_train.bat again; if it fails again see RUNSHEET.md "What to report".
pause
exit /b %RC%
