@echo off
rem =====================================================================
rem Step 1: One-time setup for Binaya's laptop (RTX 3050 GPU)
rem =====================================================================
setlocal
cd /d "%~dp0"

echo [1/4] Checking Python installation...
set PY=
py -3.12 -c "import sys" >nul 2>&1 && set PY=py -3.12
if not defined PY py -3.11 -c "import sys" >nul 2>&1 && set PY=py -3.11
if not defined PY py -3.10 -c "import sys" >nul 2>&1 && set PY=py -3.10
if not defined PY py -3.13 -c "import sys" >nul 2>&1 && set PY=py -3.13
if not defined PY python -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,14) else 1)" >nul 2>&1 && set PY=python

if not defined PY (
  echo.
  echo [ERROR] Python 3.10, 3.11, or 3.12 was not found.
  echo Please download Python 3.12 from:
  echo   https://www.python.org/downloads/windows/
  echo IMPORTANT: During installation, make sure to CHECK the box:
  echo   [X] Add python.exe to PATH
  echo After installing Python, double-click this 1_setup.bat file again.
  echo.
  pause
  exit /b 1
)
echo Found Python: %PY%

echo.
echo [2/4] Creating local virtual environment (venv)...
if not exist venv\Scripts\python.exe (
  %PY% -m venv venv || goto :fail
)

echo.
echo [3/4] Installing PyTorch with CUDA support for RTX 3050...
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu124 || venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu121 || goto :fail
venv\Scripts\python -m pip install numpy numba kaggle || goto :fail

echo.
echo [4/4] Verifying GPU and dependencies...
echo =====================================================================
venv\Scripts\python check_env.py
echo =====================================================================
echo.
echo Setup completed! You can now run 2_test_gpu.bat or 3_run_training.bat.
pause
exit /b 0

:fail
echo.
echo [ERROR] Something failed during installation.
echo Please take a screenshot or copy the error message and send it to Shishir.
pause
exit /b 1
