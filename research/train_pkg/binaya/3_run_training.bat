@echo off
rem =====================================================================
rem Step 3: Run Model Training on RTX 3050
rem Safe to stop anytime by closing the window. Re-opening resumes automatically!
rem =====================================================================
setlocal
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
  echo [ERROR] Environment not found. Please double-click 1_setup.bat first!
  pause
  exit /b 1
)

if not exist meta.json (
  echo [ERROR] Training data files are missing from this folder.
  echo Please make sure the .npy data files are copied into this folder.
  pause
  exit /b 1
)

echo.
echo =====================================================================
echo  STARTING FPNet TRAINING (RTX 3050 Optimized)
echo  Batch Size: 128 (Safe VRAM limit for RTX 3050)
echo  Job: FULL-data network, seed offset 50 (fulls50)
echo =====================================================================
echo.

venv\Scripts\python train_fp.py --data . --out out_binaya --seed_offset 50 --tag fulls50 --bs 128 %*
set RC=%ERRORLEVEL%

echo.
if %RC%==0 (
  echo =====================================================================
  echo  TRAINING COMPLETE!
  echo  Now double-click 4_send_results.bat to send the models.
  echo =====================================================================
  venv\Scripts\python check_result.py --data . --out out_binaya
) else if %RC%==2 (
  echo Training paused or interrupted. Double-click 3_run_training.bat again to continue.
) else if %RC%==3 (
  echo OUT OF GPU MEMORY. Please close games or other apps and run again.
) else (
  echo Training stopped with error code %RC%.
  echo Double-click 3_run_training.bat to resume, or take a screenshot and send to Shishir.
)

pause
exit /b %RC%
