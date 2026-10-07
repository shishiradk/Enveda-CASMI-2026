@echo off
rem =====================================================================
rem Step 2: Quick 30-second GPU test for RTX 3050
rem =====================================================================
setlocal
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
  echo [ERROR] Environment not found. Please double-click 1_setup.bat first!
  pause
  exit /b 1
)

echo Checking GPU availability and running test...
venv\Scripts\python -c "
import torch
print('PyTorch Version :', torch.__version__)
print('CUDA Available  :', torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print('GPU Device Name :', props.name)
    print('VRAM Total      :', f'{props.total_memory / 1e9:.2f} GB')
    print('Running matrix multiplication test on GPU...')
    a = torch.randn(2048, 2048, device='cuda')
    b = (a @ a).sum().item()
    print('SUCCESS: Your GPU is working perfectly!')
else:
    print('WARNING: CUDA is not available. Please send a screenshot to Shishir.')
"
echo.
pause
