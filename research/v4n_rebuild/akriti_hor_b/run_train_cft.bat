@echo off
rem Starts the CFT training, or continues it from the last checkpoint if it was interrupted.
rem Just run this file again after any interruption. Optional: run_train_cft.bat --bs 128
setlocal
cd /d "%~dp0"
call "%~dp0_cft_env.bat"
if errorlevel 1 (
  pause
  exit /b 1
)
%PYEXE% train_cft.py --data %DATA% --out out_cft --name hoR_b --seed 2 --merge_p 0.6 %*
set RC=%ERRORLEVEL%
echo.
if %RC%==0 echo FINISHED. Now run check_result_cft.bat
if %RC%==2 echo NOT FINISHED YET. Run run_train_cft.bat again to continue.
if %RC%==3 echo OUT OF GPU MEMORY. Run:  run_train_cft.bat --bs 128
if not %RC%==0 if not %RC%==2 if not %RC%==3 echo STOPPED WITH AN ERROR. Run run_train_cft.bat again; if it fails again see RUNSHEET_CFT.md "What to report".
pause
exit /b %RC%
