@echo off
rem Evaluates a pair of fingerprint networks (fp_single*.pt + fp_merged*.pt) on the four proxy scenarios.
rem   run_bench.bat                                asks for the folder and a name
rem   run_bench.bat <folder> <name> [workers]      e.g.  run_bench.bat C:\casmi-train-pkg\out\export ho1
rem Run the same command again after an interruption: finished scenarios are kept, the rest runs again.
setlocal
cd /d "%~dp0"
set "MODELS=%~1"
set "TAG=%~2"
set "WORKERS=%~3"
if "%MODELS%"=="" set /p MODELS=Folder that contains the fp_single / fp_merged .pt files:
if "%TAG%"=="" set /p TAG=Short name for this run, no spaces (for example ho1):
if "%WORKERS%"=="" set WORKERS=3
rem "Copy as path" in Explorer adds quotes: remove them.
set "MODELS=%MODELS:"=%"
if "%TAG%"=="" (
  echo PROBLEM: the run name is empty. Run run_bench.bat again and type a name such as ho1.
  pause
  exit /b 1
)
if not "%TAG%"=="%TAG: =%" (
  echo PROBLEM: the run name "%TAG%" contains a space. Use letters, digits, - or _ only.
  pause
  exit /b 1
)
if not exist "%MODELS%\*merged*.pt" (
  echo PROBLEM: no *merged*.pt file in "%MODELS%".
  pause
  exit /b 1
)
set TODO=
for /f %%s in ('venv\Scripts\python bench_check.py todo %TAG%') do set TODO=%%s
if "%TODO%"=="" goto :fail
echo Scenarios to run: %TODO%
if not "%TODO%"=="none" (
  venv\Scripts\python research\bench\bench.py run --tag %TAG% --scen %TODO% --alt-fp "%MODELS%" --workers %WORKERS% || goto :fail
)
venv\Scripts\python research\bench\bench.py eval --tag %TAG% > results\bench\%TAG%_eval.txt || goto :fail
echo.
venv\Scripts\python bench_check.py full %TAG% || goto :fail
echo.
echo FINISHED. Now run send_bench_results.bat
pause
exit /b 0

:fail
echo.
echo PROBLEM: the bench stopped with an error. Run run_bench.bat again with the same folder and name;
echo if it fails the same way, report it (see RUNSHEET_BENCH.md, "What to report").
pause
exit /b 1
