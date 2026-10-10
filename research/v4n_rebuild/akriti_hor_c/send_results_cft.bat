@echo off
rem Uploads out_cft\export as a PRIVATE Kaggle dataset "casmi-cft-hor-c-results" under your account.
rem Needs your Kaggle API token: kaggle.com -> Settings -> API -> Create New Token,
rem then put the downloaded kaggle.json into  %USERPROFILE%\.kaggle\
setlocal
cd /d "%~dp0"
call "%~dp0_cft_env.bat"
if errorlevel 1 (
  pause
  exit /b 1
)
if not exist out_cft\export\check_summary_cft.txt (
  echo PROBLEM: out_cft\export\check_summary_cft.txt not found. Run run_train_cft.bat and check_result_cft.bat first.
  pause
  exit /b 1
)
set /p KUSER=Type your Kaggle username and press Enter: 
> out_cft\export\dataset-metadata.json echo {"title": "casmi-cft-hor-c-results", "id": "%KUSER%/casmi-cft-hor-c-results", "licenses": [{"name": "other"}]}
%KAGGLE% datasets create -p out_cft\export
if errorlevel 1 (
  echo.
  echo If it says the dataset already exists, run this instead:
  echo   %KAGGLE% datasets version -p out_cft\export -m "update"
)
echo.
echo Now open https://www.kaggle.com/datasets/%KUSER%/casmi-cft-hor-c-results  -^> Settings -^> Sharing
echo and add  shishiradhikari11  as a collaborator ^(keep the dataset private^).
pause
