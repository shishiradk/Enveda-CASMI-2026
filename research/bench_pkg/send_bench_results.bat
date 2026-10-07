@echo off
rem Uploads results\bench\<name> (about 100 MB) as a PRIVATE Kaggle dataset "casmi-bench-<name>" under your account.
setlocal
cd /d "%~dp0"
set /p TAG=Name of the run you want to send (the one you typed in run_bench.bat):
set /p KUSER=Your Kaggle username:
if not exist results\bench\%TAG%\bench_summary.txt (
  echo PROBLEM: results\bench\%TAG%\bench_summary.txt not found. Run run_bench.bat first.
  pause
  exit /b 1
)
> results\bench\%TAG%\dataset-metadata.json echo {"title": "casmi-bench-%TAG%", "id": "%KUSER%/casmi-bench-%TAG%", "licenses": [{"name": "other"}]}
venv\Scripts\kaggle datasets create -p results\bench\%TAG%
if errorlevel 1 (
  echo.
  echo If it says the dataset already exists, run this instead:
  echo   venv\Scripts\kaggle datasets version -p results\bench\%TAG% -m "update"
)
echo.
echo Now open https://www.kaggle.com/datasets/%KUSER%/casmi-bench-%TAG%  -^> Settings -^> Sharing
echo and add  shishiradhikari11  as a collaborator ^(keep the dataset private^).
echo Then paste the text of results\bench\%TAG%\bench_summary.txt into the team chat.
pause
