@echo off
rem Downloads the two ho2 datasets file by file into THIS folder (use when Kaggle's "Download" button says
rem "Data not available"). Safe to run again: files that are already complete are skipped.
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set K=
if exist "C:\casmi-train-pkg\venv\Scripts\kaggle.exe" set K="C:\casmi-train-pkg\venv\Scripts\kaggle.exe"
if not defined K (where kaggle >nul 2>nul && set K=kaggle)
if not defined K (
  echo Installing the Kaggle tool ...
  py -m pip install --user kaggle || python -m pip install --user kaggle
  set K=py -m kaggle.cli
)
if not exist "%USERPROFILE%\.kaggle\kaggle.json" (
  echo PROBLEM: %USERPROFILE%\.kaggle\kaggle.json not found. Kaggle website: Settings - API - Create New Token,
  echo and put the downloaded kaggle.json into %USERPROFILE%\.kaggle\
  pause
  exit /b 1
)
set BAD=0
set N=0
call :get binayaadhikari13/casmi-train-pkg-full check_result.py 4210
call :get binayaadhikari13/casmi-train-pkg-full fp_bits.npy 55568
call :get binayaadhikari13/casmi-train-pkg-full fp_model.py 6120
call :get binayaadhikari13/casmi-train-pkg-full meta.json 3380
call :get binayaadhikari13/casmi-train-pkg-full mol_fold.npy 275938
call :get binayaadhikari13/casmi-train-pkg-full mol_key.npy 3861468
call :get binayaadhikari13/casmi-train-pkg-full mol_pool.npy 1103368
call :get binayaadhikari13/casmi-train-pkg-full mol_smiles.tsv 16606229
call :get binayaadhikari13/casmi-train-pkg-full mol_split.npy 275938
call :get binayaadhikari13/casmi-train-pkg-full pool_fp.npy 617476661
call :get binayaadhikari13/casmi-train-pkg-full pool_key.npy 9970914
call :get binayaadhikari13/casmi-train-pkg-full pool_mass.npy 5697720
call :get binayaadhikari13/casmi-train-pkg-full pool_smiles.txt 44992739
call :get binayaadhikari13/casmi-train-pkg-full pool_src.npy 712327
call :get binayaadhikari13/casmi-train-pkg-full spec_ad.npy 2528360
call :get binayaadhikari13/casmi-train-pkg-full spec_ce.npy 10113056
call :get binayaadhikari13/casmi-train-pkg-full spec_ins.npy 2528360
call :get binayaadhikari13/casmi-train-pkg-full spec_it.npy 428575792
call :get binayaadhikari13/casmi-train-pkg-full spec_lib.npy 2528360
call :get binayaadhikari13/casmi-train-pkg-full spec_mode.npy 2528360
call :get binayaadhikari13/casmi-train-pkg-full spec_mol.npy 10113056
call :get binayaadhikari13/casmi-train-pkg-full spec_mz.npy 428575792
call :get binayaadhikari13/casmi-train-pkg-full spec_off.npy 20225992
call :get binayaadhikari13/casmi-train-pkg-full spec_prec.npy 10113056
call :get binayaadhikari13/casmi-train-pkg-full spec_row.npy 10113056
call :get binayaadhikari13/casmi-train-pkg-full spec_split.npy 2528360
call :get binayaadhikari13/casmi-train-cft-full cfp_bits.npy 89720
call :get binayaadhikari13/casmi-train-cft-full cft_formulas.txt 1909582
call :get binayaadhikari13/casmi-train-cft-full cft_fp.py 3689
call :get binayaadhikari13/casmi-train-cft-full cft_meta.json 3209
call :get binayaadhikari13/casmi-train-cft-full cft_model.py 14344
call :get binayaadhikari13/casmi-train-cft-full cpool_elem.npy 11268948
call :get binayaadhikari13/casmi-train-cft-full cpool_form.npy 4507656
call :get binayaadhikari13/casmi-train-cft-full cpool_fp.npy 1577634928
call :get binayaadhikari13/casmi-train-cft-full cpool_key.npy 15776476
call :get binayaadhikari13/casmi-train-cft-full cpool_kid.npy 4507656
call :get binayaadhikari13/casmi-train-cft-full cpool_mass.npy 9015184
call :get binayaadhikari13/casmi-train-cft-full cpool_orig.npy 4507656
call :get binayaadhikari13/casmi-train-cft-full cpool_smiles.txt 65329534
call :get binayaadhikari13/casmi-train-cft-full cpool_src.npy 1127010
call :get binayaadhikari13/casmi-train-cft-full eval_cft.py 10881
call :get binayaadhikari13/casmi-train-cft-full mol_alt.npy 1103368
call :get binayaadhikari13/casmi-train-cft-full mol_cpool.npy 1103368
call :get binayaadhikari13/casmi-train-cft-full pcv_form.npy 744032
call :get binayaadhikari13/casmi-train-cft-full pcv_fp.npy 260366528
call :get binayaadhikari13/casmi-train-cft-full pcv_fpb.npy 161241320
call :get binayaadhikari13/casmi-train-cft-full pcv_mol.npy 2128
call :get binayaadhikari13/casmi-train-cft-full pcv_nfull.npy 4128
call :get binayaadhikari13/casmi-train-cft-full pcv_off.npy 4136
call :get binayaadhikari13/casmi-train-cft-full pcv_smiles.txt 9138326
call :get binayaadhikari13/casmi-train-cft-full pcv_truth_fp.npy 700128
call :get binayaadhikari13/casmi-train-cft-full pcv_truth_fpb.npy 433628
call :get binayaadhikari13/casmi-train-cft-full pcv_truth_in_pc.npy 628
call :get binayaadhikari13/casmi-train-cft-full train_cft.py 39943
echo.
if !BAD!==0 (echo ALL FILES DOWNLOADED - %N% files. Next: double-click setup_windows.bat) else (echo !BAD! file^(s^) failed. Run download_full.bat again.)
pause
exit /b 0

:get
set /a N+=1
if exist "%~2" if "%~z2"=="%3" (echo ok   %~2 & exit /b 0)
echo get  %~2 ...
%K% datasets download %1 -f %2 -p . >nul 2>nul
if exist "%~2.zip" (powershell -NoProfile -Command "Expand-Archive -Force '%~2.zip' '.'" & del "%~2.zip")
if exist "%~2" if "%~z2"=="%3" (echo ok   %~2 & exit /b 0)
echo FAIL %~2
set /a BAD+=1
exit /b 0
