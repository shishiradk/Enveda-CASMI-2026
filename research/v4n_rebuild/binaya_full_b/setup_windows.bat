@echo off
rem One-time setup: creates a private Python environment in the "venv" folder and installs PyTorch with CUDA.
setlocal
cd /d "%~dp0"

set PY=
py -3.12 -c "import sys" >nul 2>&1 && set PY=py -3.12
if not defined PY py -3.13 -c "import sys" >nul 2>&1 && set PY=py -3.13
if not defined PY py -3.11 -c "import sys" >nul 2>&1 && set PY=py -3.11
if not defined PY py -3.10 -c "import sys" >nul 2>&1 && set PY=py -3.10
if not defined PY python -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,14) else 1)" >nul 2>&1 && set PY=python
if not defined PY (
  echo.
  echo PROBLEM: Python 3.10 - 3.14 was not found.
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
rem PyTorch 2.11.0 built for CUDA 12.8 (needed for RTX 50-series cards). About 3 GB download.
venv\Scripts\python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 || goto :fail
venv\Scripts\python -m pip install numpy numba kaggle || goto :fail

echo.
echo ================== CHECK ==================
venv\Scripts\python check_env.py
echo ===========================================
pause
exit /b 0

:fail
echo.
echo PROBLEM: an install step failed. Copy the text above and send it back (see RUNSHEET.md, "What to report").
pause
exit /b 1
