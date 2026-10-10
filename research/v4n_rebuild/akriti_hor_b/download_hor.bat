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
call :get akritirijal04/casmi-train-pkg-hor check_result.py 4210
call :get akritirijal04/casmi-train-pkg-hor fp_bits.npy 55568
call :get akritirijal04/casmi-train-pkg-hor fp_model.py 6120
call :get akritirijal04/casmi-train-pkg-hor meta.json 11809
call :get akritirijal04/casmi-train-pkg-hor mol_fold.npy 275938
call :get akritirijal04/casmi-train-pkg-hor mol_key.npy 3861468
call :get akritirijal04/casmi-train-pkg-hor mol_pool.npy 1103368
call :get akritirijal04/casmi-train-pkg-hor mol_smiles.tsv 16606229
call :get akritirijal04/casmi-train-pkg-hor mol_split.npy 275938
call :get akritirijal04/casmi-train-pkg-hor pool_fp.npy 598135625
call :get akritirijal04/casmi-train-pkg-hor pool_key.npy 9658602
call :get akritirijal04/casmi-train-pkg-hor pool_mass.npy 5519256
call :get akritirijal04/casmi-train-pkg-hor pool_smiles.txt 43979788
call :get akritirijal04/casmi-train-pkg-hor pool_src.npy 690019
call :get akritirijal04/casmi-train-pkg-hor spec_ad.npy 2225401
call :get akritirijal04/casmi-train-pkg-hor spec_ce.npy 8901220
call :get akritirijal04/casmi-train-pkg-hor spec_ins.npy 2225401
call :get akritirijal04/casmi-train-pkg-hor spec_it.npy 381852536
call :get akritirijal04/casmi-train-pkg-hor spec_lib.npy 2225401
call :get akritirijal04/casmi-train-pkg-hor spec_mode.npy 2225401
call :get akritirijal04/casmi-train-pkg-hor spec_mol.npy 8901220
call :get akritirijal04/casmi-train-pkg-hor spec_mz.npy 381852536
call :get akritirijal04/casmi-train-pkg-hor spec_off.npy 17802320
call :get akritirijal04/casmi-train-pkg-hor spec_prec.npy 8901220
call :get akritirijal04/casmi-train-pkg-hor spec_row.npy 8901220
call :get akritirijal04/casmi-train-pkg-hor spec_split.npy 2225401
call :get akritirijal04/casmi-train-cft-hor cfp_bits.npy 89496
call :get akritirijal04/casmi-train-cft-hor cft_formulas.txt 1864873
call :get akritirijal04/casmi-train-cft-hor cft_fp.py 3689
call :get akritirijal04/casmi-train-cft-hor cft_meta.json 3223
call :get akritirijal04/casmi-train-cft-hor cft_model.py 14344
call :get akritirijal04/casmi-train-cft-hor cpool_elem.npy 11035898
call :get akritirijal04/casmi-train-cft-hor cpool_form.npy 4414436
call :get akritirijal04/casmi-train-cft-hor cpool_fp.npy 1541697197
call :get akritirijal04/casmi-train-cft-hor cpool_key.npy 15450206
call :get akritirijal04/casmi-train-cft-hor cpool_kid.npy 4414436
call :get akritirijal04/casmi-train-cft-hor cpool_mass.npy 8828744
call :get akritirijal04/casmi-train-cft-hor cpool_orig.npy 4414436
call :get akritirijal04/casmi-train-cft-hor cpool_smiles.txt 64185406
call :get akritirijal04/casmi-train-cft-hor cpool_src.npy 1103705
call :get akritirijal04/casmi-train-cft-hor eval_cft.py 10881
call :get akritirijal04/casmi-train-cft-hor mol_alt.npy 1103368
call :get akritirijal04/casmi-train-cft-hor mol_cpool.npy 1103368
call :get akritirijal04/casmi-train-cft-hor pcv_form.npy 756956
call :get akritirijal04/casmi-train-cft-hor pcv_fp.npy 264322307
call :get akritirijal04/casmi-train-cft-hor pcv_fpb.npy 164042597
call :get akritirijal04/casmi-train-cft-hor pcv_mol.npy 2128
call :get akritirijal04/casmi-train-cft-hor pcv_nfull.npy 4128
call :get akritirijal04/casmi-train-cft-hor pcv_off.npy 4136
call :get akritirijal04/casmi-train-cft-hor pcv_smiles.txt 9192090
call :get akritirijal04/casmi-train-cft-hor pcv_truth_fp.npy 698628
call :get akritirijal04/casmi-train-cft-hor pcv_truth_fpb.npy 433628
call :get akritirijal04/casmi-train-cft-hor pcv_truth_in_pc.npy 628
call :get akritirijal04/casmi-train-cft-hor train_cft.py 39943
echo.
if !BAD!==0 (echo ALL FILES DOWNLOADED - %N% files. Next: double-click setup_windows.bat) else (echo !BAD! file^(s^) failed. Run download_hor.bat again.)
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
