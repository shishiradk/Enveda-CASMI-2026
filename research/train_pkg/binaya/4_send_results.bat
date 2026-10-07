@echo off
rem =====================================================================
rem Step 4: Package results to send to Shishir
rem =====================================================================
setlocal
cd /d "%~dp0"

if not exist out_binaya\export (
  echo [ERROR] No exported models found in out_binaya\export.
  echo Please make sure training finished completely.
  pause
  exit /b 1
)

if exist binaya_results rmdir /s /q binaya_results
mkdir binaya_results

echo Copying trained weights and summaries...
copy /y out_binaya\export\fp_*.pt binaya_results\ >nul
copy /y out_binaya\export\train_result.json binaya_results\ >nul
copy /y out_binaya\export\check_summary.txt binaya_results\ >nul

echo.
echo =====================================================================
echo  FILES READY in folder: binaya_results\
echo =====================================================================
dir binaya_results\
echo.
echo Please send the files in the 'binaya_results' folder (or zip the folder)
echo to Shishir via Google Drive, Telegram, or WhatsApp.
echo.
pause
