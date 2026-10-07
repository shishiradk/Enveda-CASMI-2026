@echo off
rem Optional 2-5 minute test of the whole pipeline (tiny run, results are thrown-away quality).
setlocal
cd /d "%~dp0"
venv\Scripts\python train_fp.py --data . --out out_smoke --smoke --fresh
venv\Scripts\python check_result.py --data . --out out_smoke
pause
