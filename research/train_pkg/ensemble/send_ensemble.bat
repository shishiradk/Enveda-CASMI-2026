@echo off
rem Uploads the 4 finished jobs as ONE private Kaggle dataset "casmi-fp-ens-ho1".
setlocal
cd /d "%~dp0"
if exist ens_upload rmdir /s /q ens_upload
mkdir ens_upload
for %%D in (out_s10 out_s20 out_s30 out_big) do (
  if exist %%D\export\train_result.json (
    copy /y %%D\export\fp_*.pt ens_upload\ >nul
    copy /y %%D\export\train_result.json ens_upload\%%D_train_result.json >nul
    copy /y %%D\export\check_summary.txt ens_upload\%%D_check_summary.txt >nul
  ) else (
    echo NOTE: %%D has no results yet - it will be missing from the upload.
  )
)
set /p KUSER=Type your Kaggle username and press Enter: 
> ens_upload\dataset-metadata.json echo {"title": "casmi-fp-ens-ho1", "id": "%KUSER%/casmi-fp-ens-ho1", "licenses": [{"name": "other"}]}
dir ens_upload
venv\Scripts\kaggle datasets create -p ens_upload
if errorlevel 1 (
  echo.
  echo If it says the dataset already exists, run this instead:
  echo   venv\Scripts\kaggle datasets version -p ens_upload -m "update"
)
echo.
echo Now open https://www.kaggle.com/datasets/%KUSER%/casmi-fp-ens-ho1  -^> Settings -^> Sharing
echo and add  shishiradhikari11  as a collaborator ^(keep it private^).
echo Then send Shishir the text of the 4 files ens_upload\*_check_summary.txt ^(a photo is fine^).
pause
