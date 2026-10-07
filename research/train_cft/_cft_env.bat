@echo off
rem Shared by the CFT .bat files: finds the Python environment and the warm-up data.
rem Works when the casmi-train-cft files are in the SAME folder as casmi-train-pkg, or in a folder next to it.
set PYEXE=
set KAGGLE=
set DATA=
if exist venv\Scripts\python.exe set PYEXE=venv\Scripts\python.exe
if exist venv\Scripts\python.exe set KAGGLE=venv\Scripts\kaggle
if not defined PYEXE if exist ..\casmi-train-pkg\venv\Scripts\python.exe set PYEXE=..\casmi-train-pkg\venv\Scripts\python.exe
if not defined KAGGLE set KAGGLE=..\casmi-train-pkg\venv\Scripts\kaggle
if exist spec_mz.npy set DATA=.
if not defined DATA if exist ..\casmi-train-pkg\spec_mz.npy set DATA=. ..\casmi-train-pkg
if not defined PYEXE (
  echo PROBLEM: the Python environment was not found. Put these files into the casmi-train-pkg folder
  echo ^(or next to it^) and run setup_windows.bat there first.
  exit /b 1
)
if not defined DATA (
  echo PROBLEM: spec_mz.npy was not found. Put these files into the casmi-train-pkg folder ^(or next to it^).
  exit /b 1
)
if not exist cpool_fp.npy (
  echo PROBLEM: cpool_fp.npy is missing - the casmi-train-cft download is incomplete.
  exit /b 1
)
exit /b 0
