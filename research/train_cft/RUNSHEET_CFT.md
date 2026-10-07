# RUNSHEET - train the CFT model (second job, after the warm-up)

Same procedure as the warm-up job (`RUNSHEET.md`); only the file names differ. One network is trained
("CFT", about 50 M parameters). Rough guess: 3-8 hours on the GPU (nobody has timed it yet).

You need BOTH private datasets: `casmi-train-pkg` (you already have it) and the new `casmi-train-cft` (about 2.1 GB).

Choose ONE way: **A. Windows PC (RTX 5050)** or **B. Kaggle notebook (T4)**.

---

## A. Windows PC

Before you start: the warm-up setup (`setup_windows.bat`) must have printed `SETUP OK` once; about **6 GB free
disk**; sleep turned off (see `RUNSHEET.md`). The warm-up job must not be running at the same time.

1. **Get the new files**: download https://www.kaggle.com/datasets/shishiradhikari11/casmi-train-cft and unzip it
   **into the same folder as the warm-up package** (for example `C:\casmi-train-pkg`), so that `train_cft.py`,
   `cpool_fp.npy` and `train_fp.py`, `spec_mz.npy` sit side by side. Command line:
   ```
   kaggle datasets download shishiradhikari11/casmi-train-cft -p C:\casmi-train-pkg --unzip
   ```
   (A separate folder next to it, named anything, also works as long as the warm-up folder is called
   `casmi-train-pkg`.)
2. Optional, 3-5 minutes: double-click `smoke_test_cft.bat`. It must end with `RESULT: ALL OK`.
3. **Double-click `run_train_cft.bat`.** Lines like `[default] step 200 ctr ... step/s vram ...G` appear, and every
   2,000 steps a line `VAL step ... pool MRR ...`. It ends with **`ALL DONE`** / `FINISHED`.
4. **Double-click `check_result_cft.bat`** (a few minutes). It must end with **`RESULT: ALL OK`**.
5. **Send back**: double-click `send_results_cft.bat` and type your Kaggle username (uploads `out_cft\export`,
   about 200 MB, as a private dataset; then add **shishiradhikari11** under Settings -> Sharing).
   Paste the text of `out_cft\export\check_summary_cft.txt` into the team chat.

The one command behind `run_train_cft.bat` (also works on Linux):
`python train_cft.py --data . --out out_cft`

**Please report early** (after about 15 minutes of training): the `step/s` and `vram` numbers of one log line.

### If it was interrupted
Double-click **`run_train_cft.bat` again**. It continues from the last checkpoint (saved every 10 minutes).
Do not delete the `out_cft` folder.

### What to do if ...
| Problem | Do this |
|---|---|
| `OUT OF GPU MEMORY` | Close programs using the GPU, then run `run_train_cft.bat --bs 128` from a Command Prompt in the folder (use `--bs 128` every time from then on; tell us). |
| `PROBLEM: the Python environment was not found` | The files are not in (or next to) the `casmi-train-pkg` folder, or `setup_windows.bat` was never run there. |
| It is very slow (below 1 step/s) | The first lines must say `device=cuda`. If they say `device=cpu`, see `RUNSHEET.md` ("torch.cuda.is_available(): False"). If it is cuda and still slow, report the `step/s` value; we may switch to `--preset small`. |
| PC has only little free memory / swaps | Add `--mmap` : `run_train_cft.bat --mmap` |
| `non-finite loss ... batch skipped` on every line | Stop and report. A few such lines are harmless. |
| Anything else | Run it once more. If it fails the same way, report (below). |

### What to report if it fails
(1) `out_cft\train_log.txt`, (2) the last 30 lines in the window, (3) which step of this sheet you were on.

### Options (only if we ask for them)
`run_train_cft.bat --preset small` trains the small model (files are then named `..._small...`; the two models do not
overwrite each other). `--steps 60000` trains longer. `--seed 2` trains a second copy (add `--name default_s2`).

---

## B. Kaggle notebook (T4)

1. kaggle.com -> Create -> New Notebook -> File -> **Import Notebook** -> upload `casmi_cft_train.ipynb`
   (it is in the `casmi-train-cft` folder).
2. **Add Input** -> add BOTH `casmi-train-pkg` and `casmi-train-cft`. If the warm-up notebook has finished, also
   add its output (Add Input -> Notebook Output Files): the evaluation then compares against the warm-up networks.
   **Settings -> Accelerator -> GPU T4.** Internet off.
3. **Save Version -> Save & Run All (Commit).** Limit: 12 hours per run; the notebook stops training after
   10 hours and evaluates.
4. When done: open the version -> **Output** -> download `out_cft/export`, or share the notebook with
   **shishiradhikari11**.
5. If the log ends with `NOT FINISHED`: edit the notebook -> Add Input -> Notebook Output Files -> this same
   notebook -> Save & Run All again. It resumes from the checkpoint.

Command-line alternative: copy `casmi_cft_train.ipynb` and `kernel-metadata-cft.json` into an empty folder, rename
the latter to `kernel-metadata.json`, replace `KAGGLE_USERNAME`, then `kaggle kernels push -p <folder>`.

Report on failure: the notebook log (Logs tab -> download) and the version number.
