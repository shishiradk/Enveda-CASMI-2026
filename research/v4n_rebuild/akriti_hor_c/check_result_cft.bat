@echo off
rem Evaluates the exported CFT model against the baselines and writes out_cft\export\check_summary_cft.txt
rem (send this text back). Takes a few minutes on the GPU.
setlocal
cd /d "%~dp0"
call "%~dp0_cft_env.bat"
if errorlevel 1 (
  pause
  exit /b 1
)
%PYEXE% eval_cft.py --data %DATA% --out out_cft %*
echo.
echo The summary above is saved in out_cft\export\check_summary_cft.txt
pause
