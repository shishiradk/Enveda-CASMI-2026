@echo off
rem Uploads out\export as a PRIVATE Kaggle dataset "casmi-fp-ho1-results" under your account.
rem Needs your Kaggle API token: kaggle.com -> Settings -> API -> Create New Token,
rem then put the downloaded kaggle.json into  %USERPROFILE%\.kaggle\
setlocal
cd /d "%~dp0"
if not exist out\export\train_result.json (
  echo PROBLEM: out\export\train_result.json not found. Run run_train.bat and check_result.bat first.
  pause
  exit /b 1
)
set /p KUSER=Type your Kaggle username and press Enter: 
> out\export\dataset-metadata.json echo {"title": "casmi-fp-ho1-results", "id": "%KUSER%/casmi-fp-ho1-results", "licenses": [{"name": "other"}]}
venv\Scripts\kaggle datasets create -p out\export
if errorlevel 1 (
  echo.
  echo If it says the dataset already exists, run this instead:
  echo   venv\Scripts\kaggle datasets version -p out\export -m "update"
)
echo.
echo Now open https://www.kaggle.com/datasets/%KUSER%/casmi-fp-ho1-results  -^> Settings -^> Sharing
echo and add  shishiradhikari11  as a collaborator ^(keep the dataset private^).
pause
