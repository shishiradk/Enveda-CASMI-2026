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
call :get akritirijal04/casmi-train-pkg-ho2 casmi_fp_train.ipynb 2412
call :get akritirijal04/casmi-train-pkg-ho2 check_env.py 1978
call :get akritirijal04/casmi-train-pkg-ho2 check_result.bat 270
call :get akritirijal04/casmi-train-pkg-ho2 check_result.py 4210
call :get akritirijal04/casmi-train-pkg-ho2 fp_bits.npy 55568
call :get akritirijal04/casmi-train-pkg-ho2 fp_model.py 6120
call :get akritirijal04/casmi-train-pkg-ho2 meta.json 3853
call :get akritirijal04/casmi-train-pkg-ho2 mol_fold.npy 275938
call :get akritirijal04/casmi-train-pkg-ho2 mol_key.npy 3861468
call :get akritirijal04/casmi-train-pkg-ho2 mol_pool.npy 1103368
call :get akritirijal04/casmi-train-pkg-ho2 mol_smiles.tsv 16606229
call :get akritirijal04/casmi-train-pkg-ho2 mol_split.npy 275938
call :get akritirijal04/casmi-train-pkg-ho2 pool_fp.npy 616459670
call :get akritirijal04/casmi-train-pkg-ho2 pool_key.npy 9954492
call :get akritirijal04/casmi-train-pkg-ho2 pool_mass.npy 5688336
call :get akritirijal04/casmi-train-pkg-ho2 pool_smiles.txt 44941916
call :get akritirijal04/casmi-train-pkg-ho2 pool_src.npy 711154
call :get akritirijal04/casmi-train-pkg-ho2 PROMPT_FOR_ASSISTANT.md 2583
call :get akritirijal04/casmi-train-pkg-ho2 run_train.bat 768
call :get akritirijal04/casmi-train-pkg-ho2 RUNSHEET.md 5768
call :get akritirijal04/casmi-train-pkg-ho2 send_results.bat 1081
call :get akritirijal04/casmi-train-pkg-ho2 setup_windows.bat 1578
call :get akritirijal04/casmi-train-pkg-ho2 smoke_test.bat 276
call :get akritirijal04/casmi-train-pkg-ho2 spec_ad.npy 2465163
call :get akritirijal04/casmi-train-pkg-ho2 spec_ce.npy 9860268
call :get akritirijal04/casmi-train-pkg-ho2 spec_ins.npy 2465163
call :get akritirijal04/casmi-train-pkg-ho2 spec_it.npy 419456796
call :get akritirijal04/casmi-train-pkg-ho2 spec_lib.npy 2465163
call :get akritirijal04/casmi-train-pkg-ho2 spec_mode.npy 2465163
call :get akritirijal04/casmi-train-pkg-ho2 spec_mol.npy 9860268
call :get akritirijal04/casmi-train-pkg-ho2 spec_mz.npy 419456796
call :get akritirijal04/casmi-train-pkg-ho2 spec_off.npy 19720416
call :get akritirijal04/casmi-train-pkg-ho2 spec_prec.npy 9860268
call :get akritirijal04/casmi-train-pkg-ho2 spec_row.npy 9860268
call :get akritirijal04/casmi-train-pkg-ho2 spec_split.npy 2465163
call :get akritirijal04/casmi-train-pkg-ho2 train_fp.py 25279
call :get akritirijal04/casmi-train-cft-ho2 _cft_env.bat 1203
call :get akritirijal04/casmi-train-cft-ho2 AKRITI_HO2.md 4950
call :get akritirijal04/casmi-train-cft-ho2 casmi_cft_train.ipynb 3017
call :get akritirijal04/casmi-train-cft-ho2 cfp_bits.npy 89600
call :get akritirijal04/casmi-train-cft-ho2 cft_formulas.txt 1903105
call :get akritirijal04/casmi-train-cft-ho2 cft_fp.py 3689
call :get akritirijal04/casmi-train-cft-ho2 cft_meta.json 3212
call :get akritirijal04/casmi-train-cft-ho2 cft_model.py 14344
call :get akritirijal04/casmi-train-cft-ho2 check_result_cft.bat 413
call :get akritirijal04/casmi-train-cft-ho2 cpool_elem.npy 11256828
call :get akritirijal04/casmi-train-cft-ho2 cpool_form.npy 4502808
call :get akritirijal04/casmi-train-cft-ho2 cpool_fp.npy 1573686788
call :get akritirijal04/casmi-train-cft-ho2 cpool_key.npy 15759508
call :get akritirijal04/casmi-train-cft-ho2 cpool_kid.npy 4502808
call :get akritirijal04/casmi-train-cft-ho2 cpool_mass.npy 9005488
call :get akritirijal04/casmi-train-cft-ho2 cpool_orig.npy 4502808
call :get akritirijal04/casmi-train-cft-ho2 cpool_smiles.txt 65260136
call :get akritirijal04/casmi-train-cft-ho2 cpool_src.npy 1125798
call :get akritirijal04/casmi-train-cft-ho2 eval_cft.py 10881
call :get akritirijal04/casmi-train-cft-ho2 kernel-metadata-cft.json 431
call :get akritirijal04/casmi-train-cft-ho2 mol_alt.npy 1103368
call :get akritirijal04/casmi-train-cft-ho2 mol_cpool.npy 1103368
call :get akritirijal04/casmi-train-cft-ho2 pcv_form.npy 750976
call :get akritirijal04/casmi-train-cft-ho2 pcv_fp.npy 262421504
call :get akritirijal04/casmi-train-cft-ho2 pcv_fpb.npy 162746432
call :get akritirijal04/casmi-train-cft-ho2 pcv_mol.npy 2128
call :get akritirijal04/casmi-train-cft-ho2 pcv_nfull.npy 4128
call :get akritirijal04/casmi-train-cft-ho2 pcv_off.npy 4136
call :get akritirijal04/casmi-train-cft-ho2 pcv_smiles.txt 9177895
call :get akritirijal04/casmi-train-cft-ho2 pcv_truth_fp.npy 699128
call :get akritirijal04/casmi-train-cft-ho2 pcv_truth_fpb.npy 433628
call :get akritirijal04/casmi-train-cft-ho2 pcv_truth_in_pc.npy 628
call :get akritirijal04/casmi-train-cft-ho2 run_train_cft.bat 764
call :get akritirijal04/casmi-train-cft-ho2 RUNSHEET_CFT.md 4864
call :get akritirijal04/casmi-train-cft-ho2 send_results_cft.bat 1173
call :get akritirijal04/casmi-train-cft-ho2 smoke_test_cft.bat 345
call :get akritirijal04/casmi-train-cft-ho2 train_cft.py 39943
echo.
if !BAD!==0 (echo ALL FILES DOWNLOADED - %N% files. Next: double-click setup_windows.bat) else (echo !BAD! file^(s^) failed. Run download_ho2.bat again.)
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
