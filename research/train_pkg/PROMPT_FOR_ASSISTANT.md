# Prompt to paste into a coding assistant (if you get stuck)

Copy everything below the line into the assistant, then paste your error message after it.

---

I am running a prepared machine-learning training package on Windows with an NVIDIA RTX 5050 (8 GB VRAM,
16 GB RAM). I am not a programmer. Help me get it running; give me exact commands to type, one at a time.

**What the package is.** One folder containing:
- data files (`spec_*.npy`, `mol_*.npy`, `pool_*.npy`, `meta.json`) - prepared mass-spectra arrays, read-only;
- `fp_model.py` - the neural network (a small transformer, PyTorch);
- `train_fp.py` - the training script. Command: `venv\Scripts\python train_fp.py --data . --out out`.
  It trains two networks one after the other ("single", then "merged"), saves a checkpoint to `out\` every
  10 minutes, resumes automatically when the same command is run again, and finally writes
  `out\export\fp_single_ho1.pt`, `out\export\fp_merged_ho1.pt` and `out\export\train_result.json`.
  Exit code 0 = finished, 2 = stopped early (run again), 3 = GPU out of memory;
- `setup_windows.bat` - creates `venv\` and installs `torch==2.11.0` from
  `https://download.pytorch.org/whl/cu128` (CUDA 12.8 build, required for RTX 50-series), plus numpy, numba, kaggle;
  then runs `check_env.py`, which must print `torch.cuda.is_available(): True` and `SETUP OK`;
- `run_train.bat` (start or resume), `check_result.bat` (verifies the exported files, must print
  `RESULT: ALL OK`), `smoke_test.bat` (short test), `send_results.bat` (uploads `out\export` to Kaggle);
- `RUNSHEET.md` - the step-by-step instructions I am following.

**You MAY:**
- fix environment problems (Python version, venv, pip, NVIDIA driver, PyTorch/CUDA build, PATH, disk space);
- lower the batch size if the GPU runs out of memory: `run_train.bat --bs 128` (then keep using it);
- lower `--workers` or add `--mmap` if system RAM is short;
- tell me how to resume, and how to collect logs.

**You MUST NOT change (the results would be unusable):**
- the model architecture or anything in `fp_model.py`;
- the data files, the train/validation split, or which spectra are used;
- the loss, learning rate, schedule, number of steps, augmentation, seeds or early-stopping settings in `train_fp.py`;
- the export format or file names in `out\export`;
- do not delete the `out` folder (it holds the checkpoints), and do not add `--fresh`.

Do not rewrite the scripts. If a code change seems necessary, stop and tell me to report the problem to my
team with `out\train_log.txt` instead.

My problem / error message:
