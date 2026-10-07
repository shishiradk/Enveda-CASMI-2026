@echo off
rem Quick test (about 5 minutes): 5 molecules, compared with the team's reference run. Must end with RESULT: ALL OK.
setlocal
cd /d "%~dp0"
if exist results\bench\smoke rmdir /s /q results\bench\smoke
venv\Scripts\python research\bench\bench.py run --tag smoke --scen S2 --limit 5 || goto :fail
venv\Scripts\python research\bench\bench.py eval --tag smoke > results\bench\smoke_eval.txt || goto :fail
venv\Scripts\python bench_check.py smoke smoke
pause
exit /b 0

:fail
echo.
echo PROBLEM: the smoke test stopped with an error. Send the text above (see RUNSHEET_BENCH.md, "What to report").
pause
exit /b 1
