@echo off
rem One-time setup for the evaluation bench: Python environment in "venv" (PyTorch with CUDA + the pinned
rem chemistry packages the caches were built with) and the competition file train.parquet.
setlocal
cd /d "%~dp0"

set PY=
py -3.12 -c "import sys" >nul 2>&1 && set PY=py -3.12
if not defined PY py -3.13 -c "import sys" >nul 2>&1 && set PY=py -3.13
if not defined PY py -3.14 -c "import sys" >nul 2>&1 && set PY=py -3.14
if not defined PY py -3.11 -c "import sys" >nul 2>&1 && set PY=py -3.11
if not defined PY python -c "import sys; sys.exit(0 if (3,11)<=sys.version_info[:2]<=(3,14) else 1)" >nul 2>&1 && set PY=python
if not defined PY (
  echo.
  echo PROBLEM: Python 3.11 - 3.14 was not found.
  echo Install Python 3.12 from https://www.python.org/downloads/windows/  ^(64-bit installer, tick "Add python.exe to PATH"^)
  echo then run this file again.
  pause
  exit /b 1
)
echo Using: %PY%

if not exist venv\Scripts\python.exe (
  %PY% -m venv venv || goto :fail
)
venv\Scripts\python -m pip install --upgrade pip || goto :fail
rem PyTorch 2.11.0 built for CUDA 12.8 (RTX 50-series), the same as the training package. About 3 GB download.
venv\Scripts\python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 || goto :fail
venv\Scripts\python -m pip install -r requirements_bench.txt || goto :fail

if exist train.parquet goto :check
echo.
echo Downloading train.parquet from the competition (about 2.9 GB) ...
venv\Scripts\kaggle competitions download -c enveda-CASMI26-molecule-id-mass-spectra -f train.parquet -p .
if exist train.parquet.zip (
  tar -xf train.parquet.zip && del train.parquet.zip
)
if not exist train.parquet (
  echo.
  echo PROBLEM: train.parquet could not be downloaded. Open
  echo   https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/rules
  echo accept the rules ^(join the competition^), then run this file again.
  echo Or copy train.parquet from the team into this folder.
  pause
  exit /b 1
)

:check
echo.
echo ================== CHECK ==================
venv\Scripts\python check_env_bench.py
echo ===========================================
pause
exit /b 0

:fail
echo.
echo PROBLEM: an install step failed. Copy the text above and send it back (see RUNSHEET_BENCH.md, "What to report").
pause
exit /b 1
