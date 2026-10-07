@echo off
rem CASMI ensemble training: 4 jobs, one after another (about 35-45 hours on an RTX 5050 laptop).
rem Safe to stop at any time (close the window / power cut): double-click this file again and it continues
rem from the last checkpoint. Finished jobs are skipped automatically.
setlocal
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
  echo PROBLEM: this folder has no venv. Put the 3 new files into C:\casmi-train-pkg, not into a new folder.
  pause
  exit /b 1
)
if not exist fp_model.py (
  echo PROBLEM: fp_model.py not found. Put the 3 new files into C:\casmi-train-pkg.
  pause
  exit /b 1
)
call :job 1 out_s10 --seed_offset 10 --tag ho1s10 || goto :stopped
call :job 2 out_s20 --seed_offset 20 --tag ho1s20 || goto :stopped
call :job 3 out_s30 --seed_offset 30 --tag ho1s30 || goto :stopped
call :job 4 out_big --seed_offset 40 --tag ho1big --d 768 --layers 8 --bs 128 || goto :stopped
echo.
echo ==================================================================
echo  ALL 4 JOBS FINISHED.  Now double-click send_ensemble.bat
echo ==================================================================
pause
exit /b 0

:job
set N=%1
set OUTD=%2
shift
shift
echo.
echo ===== JOB %N% of 4  (folder %OUTD%)  started %DATE% %TIME% =====
if exist %OUTD%\export\check_summary.txt (
  findstr /c:"RESULT: ALL OK" %OUTD%\export\check_summary.txt >nul && (echo job %N% already done - skipping & exit /b 0)
)
venv\Scripts\python train_fp.py --data . --out %OUTD% %1 %2 %3 %4 %5 %6 %7 %8 %9
set RC=%ERRORLEVEL%
if %RC%==3 (
  echo OUT OF GPU MEMORY in job %N%. Close other programs and double-click run_ensemble.bat again.
  exit /b 1
)
if not %RC%==0 (
  echo JOB %N% NOT FINISHED ^(code %RC%^). Double-click run_ensemble.bat again to continue.
  exit /b 1
)
venv\Scripts\python check_result.py --data . --out %OUTD%
exit /b 0

:stopped
echo.
echo Stopped. Double-click run_ensemble.bat again - it continues where it stopped.
echo If the SAME error appears twice, send Shishir a photo of the last 30 lines.
pause
exit /b 1
