@echo off
rem Optional 2-5 minute test of the whole pipeline (tiny run, results are thrown-away quality).
setlocal
cd /d "%~dp0"
call "%~dp0_cft_env.bat"
if errorlevel 1 (
  pause
  exit /b 1
)
%PYEXE% train_cft.py --data %DATA% --out out_cft_smoke --smoke --fresh
%PYEXE% eval_cft.py --data %DATA% --out out_cft_smoke --smoke
pause
