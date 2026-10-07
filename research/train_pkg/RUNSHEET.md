# RUNSHEET - train the two fingerprint networks (warm-up job)

You will train two small neural networks on your GPU and send back two files. Nothing here needs
programming. Total time: about 30 minutes of your attention, plus several hours of the PC working alone
(rough guess 2-6 hours; nobody has timed it on a GPU yet).

Choose ONE way: **A. your Windows PC (RTX 5050)** or **B. a Kaggle notebook (free T4 GPU)**.

---

## A. Windows PC with the RTX 5050

### Before you start
- About **10 GB free disk** (2 GB data, 6 GB PyTorch, 1.5 GB results).
- A recent NVIDIA driver (GeForce Experience / NVIDIA App -> Drivers -> update). No separate CUDA install needed.
- **Turn off sleep**: Settings -> System -> Power -> Screen and sleep -> "When plugged in, put my device to
  sleep after" = **Never**. Keep the laptop plugged in. (The screen may turn off; that is fine.)

### Steps
1. **Install Python 3.12** (skip if you have Python 3.10-3.14): https://www.python.org/downloads/windows/ ->
   64-bit installer -> tick **"Add python.exe to PATH"** -> Install.
2. **Get the package** (private Kaggle dataset, shared with you). Either
   - website: open https://www.kaggle.com/datasets/shishiradhikari11/casmi-train-pkg -> **Download** ->
     unzip into a folder, for example `C:\casmi-train-pkg`; or
   - command line (needs `pip install kaggle` and your `kaggle.json` token in `C:\Users\<you>\.kaggle\`):
     ```
     kaggle datasets download shishiradhikari11/casmi-train-pkg -p C:\casmi-train-pkg --unzip
     ```
   The folder must contain `train_fp.py`, `setup_windows.bat` and many `.npy` files, all side by side.
3. **Double-click `setup_windows.bat`.** It downloads PyTorch (about 3 GB, be patient). At the end it must print
   `torch.cuda.is_available(): True`, your GPU name and **`SETUP OK`**.
4. **Double-click `run_train.bat`.** A window shows lines like `[single] step 200 ... step/s`. Leave it running.
   It trains the first network (`single`), then the second (`merged`), and ends with **`ALL DONE`** / `FINISHED`.
5. **Double-click `check_result.bat`.** It must end with **`RESULT: ALL OK`**.
6. **Send the results back**: double-click `send_results.bat`, type your Kaggle username. It uploads the folder
   `out\export` as a private dataset. Then open the page it prints -> Settings -> Sharing -> add
   **shishiradhikari11**. (Fallback: zip `out\export` and send it any other way; it is about 290 MB.)
7. Paste the text of `out\export\check_summary.txt` into the team chat.

The one command behind `run_train.bat` (also works on Linux): `python train_fp.py --data . --out out`

### If it was interrupted (power cut, closed window, sleep, Ctrl+C)
Double-click **`run_train.bat` again**. It continues from the last checkpoint (saved every 10 minutes); you lose
at most 10 minutes. Do not delete the `out` folder.

### What you get (folder `out\export`)
`fp_single_ho1.pt`, `fp_merged_ho1.pt` (144 MB each), `train_result.json`, `train_log.txt`, `metrics.jsonl`,
`check_summary.txt`.

### What to do if ...
| Problem | Do this |
|---|---|
| `torch.cuda.is_available(): False` | Update the NVIDIA driver, restart, run `setup_windows.bat` again. Still False: open a Command Prompt in the folder and run `venv\Scripts\python -m pip install --force-reinstall torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128`, then `venv\Scripts\python check_env.py`. Last resort, same line with `torch==2.13.0` and `.../whl/cu130`. |
| "Python was not found" | Do step 1 and tick "Add python.exe to PATH"; run `setup_windows.bat` again. |
| `OUT OF GPU MEMORY` | Close games/browsers using the GPU. Open a Command Prompt in the folder and run `run_train.bat --bs 128` (use `--bs 128` every time from then on; tell us you did). |
| Disk full | Free 3 GB or more, then run `run_train.bat` again. A file ending in `.tmp` inside `out` may be deleted. |
| PC went to sleep / restarted | Turn sleep off (see above), run `run_train.bat` again. |
| It is very slow (below 1 step/s) | Check that the first lines say `device=cuda`. If they say `device=cpu`, the GPU is not used: see the first row. |
| `non-finite loss ... batch skipped` appears a few times | Harmless. If it repeats on every line, stop and report. |
| Anything else | Run it once more. If it fails the same way, report (below). |

### What to report if it fails
Send: (1) the file `out\train_log.txt`, (2) a screenshot or copy of the last 30 lines in the window,
(3) the output of `setup_windows.bat` (the CHECK block), (4) which step of this sheet you were on.

Optional quick test before the long run: `smoke_test.bat` (a few minutes on the GPU, ends with `RESULT: ALL OK`).

---

## B. Kaggle notebook (T4)

1. kaggle.com -> Create -> New Notebook -> File -> **Import Notebook** -> upload `casmi_fp_train.ipynb`
   (it is in the package folder).
2. Right panel: **Add Input** -> search `casmi-train-pkg` (Your Datasets / Shared with you) -> add.
   **Settings -> Accelerator -> GPU T4.** Internet can stay off.
3. **Save Version -> Save & Run All (Commit).** You can close the browser. Limit: 12 hours per run.
4. When it is done, open the version -> **Output** -> download the `out/export` folder, or tell us the notebook
   name and share the notebook with **shishiradhikari11** (Share -> add collaborator).
5. If the log ends with `NOT FINISHED`: edit the notebook -> Add Input -> Notebook Output Files -> pick this
   same notebook -> Save & Run All again. It resumes from the checkpoint.

Command-line alternative (in the package folder, after replacing `KAGGLE_USERNAME` in
`kernel-metadata.json`): `kaggle kernels push -p .`  then  `kaggle kernels output KAGGLE_USERNAME/casmi-fp-train-ho1 -p results`.

Report on failure: the notebook log (Logs tab -> download) and the version number.
