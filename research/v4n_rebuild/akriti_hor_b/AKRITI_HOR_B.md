# Akriti: CFT "hoR_b" training on your laptop (RTX 5050)

Hi Akriti! This is one training job for our CASMI team, on your laptop. It is the same CFT training you know, with new
data: 22,000 molecules are held out so we can test the model fairly. **You don't need to do anything on Kaggle except
download two datasets and upload the result.** Shishir runs the other job (the fingerprint networks) on Kaggle
himself. **Nothing gets submitted to the competition.**

**Time:** about 30 minutes of your attention. Then the laptop works alone for roughly **4–8 hours**.
**Before you start:** wait until Shishir messages "hoR datasets are ready".

---

## Before you start

- **Disk:** at least **15 GB free** on `C:` (data about 4 GB, Python/PyTorch about 6 GB, results about 1.5 GB).
- **Power:** keep the laptop **plugged in**.
- **Sleep off:** Settings → System → Power → Screen and sleep → "When plugged in, put my device to sleep after" =
  **Never**. The screen may turn off; that's fine.
- **Free GPU:** close games and other GPU-heavy programs. Don't run the old CFT or fingerprint training at the same
  time.

## Steps

1. **Make a new, empty folder:** `C:\casmi-hor`. Don't reuse the old `C:\casmi-train-pkg` folder; the files would
   mix.

2. **Download the two datasets into it.** Both are private datasets in your own account (Your Work → Datasets). Download each one and **unzip both
   into the same folder `C:\casmi-hor`**, so everything sits side by side:
   - https://www.kaggle.com/datasets/akritirijal04/casmi-train-pkg-hor → **Download** → unzip into
     `C:\casmi-hor`
   - https://www.kaggle.com/datasets/akritirijal04/casmi-train-cft-hor → **Download** → unzip into
     `C:\casmi-hor`

   Check: `C:\casmi-hor` now directly contains `setup_windows.bat`, `run_train_cft.bat`, `train_cft.py`,
   `spec_mz.npy` and `cpool_fp.npy` (plus many other files). If they're inside an extra sub-folder, move them up
   into `C:\casmi-hor`.

   **If Kaggle says "Data not available. This dataset may need its archive recreated"** (Kaggle sometimes fails to
   build the zip for big datasets): put **`download_hor.bat`** (it is in the zip Shishir sent) into
   `C:\casmi-hor` and double-click it.
   - It downloads all 73 files one by one (3.7 GB, roughly 10–30 minutes). It uses the Kaggle login from your earlier
     CASMI setup.
   - It must end with **`ALL FILES DOWNLOADED`**. If it says some files failed, just double-click it again: finished
     files are skipped.

   Command-line alternative (if your `kaggle.json` is set up):
   ```
   kaggle datasets download akritirijal04/casmi-train-pkg-hor -p C:\casmi-hor --unzip
   kaggle datasets download akritirijal04/casmi-train-cft-hor -p C:\casmi-hor --unzip
   ```

3. **Double-click `setup_windows.bat`.** It installs Python packages and PyTorch (about 3 GB, be patient; faster
   than last time if pip remembers the download). At the end it must print `torch.cuda.is_available(): True`, your
   GPU name and **`SETUP OK`**.

4. **Optional quick test (3–5 min):** double-click `smoke_test_cft.bat`. It must end with `RESULT: ALL OK`.

5. **Double-click `run_train_cft.bat`.** Lines like `[hoR_b] step 200 ... step/s vram ...G` appear, and every 2,000
   steps a line `VAL step ... pool MRR ...`. Leave it running. It ends with **`FINISHED`**.
   - **After about 15 minutes, please send Shishir one log line** (it shows the `step/s` and `vram` numbers).

6. **Double-click `check_result_cft.bat`** (a few minutes). It must end with **`RESULT: ALL OK`**.

7. **Send the result back:** double-click `send_results_cft.bat` and type your Kaggle username (`akritirijal04`).
   - It uploads the folder `out_cft\export` (about 200 MB) as a private dataset called `casmi-cft-hor-b-results`.
   - Then open the page it prints → **Settings → Sharing** → add **shishiradhikari11**.
   - Paste the text of `out_cft\export\check_summary_cft.txt` into the team chat.

That's all. Thank you! 🙏

---

## If something goes wrong

| What you see | What to do |
|---|---|
| The window closed, power cut, laptop slept or restarted | Double-click **`run_train_cft.bat` again**. It continues from the last checkpoint (saved every 10 minutes). **Never delete the `out_cft` folder.** |
| `OUT OF GPU MEMORY` | Close other programs. Open a Command Prompt in `C:\casmi-hor` (click the address bar of the folder window, type `cmd`, Enter), run `run_train_cft.bat --bs 128`, and **use `--bs 128` every time from then on**. Tell Shishir. |
| `torch.cuda.is_available(): False` | Update the NVIDIA driver (NVIDIA App → Drivers), restart the laptop, run `setup_windows.bat` again. Still False: tell Shishir. |
| Very slow (below 1 step/s) | Check that the first lines say `device=cuda`. If `cpu`, see the row above. If `cuda` and still slow, send Shishir the `step/s` number. |
| `PROBLEM: the Python environment was not found` | `setup_windows.bat` wasn't run in `C:\casmi-hor`, or the files are in a sub-folder (step 2). |
| `PROBLEM: cpool_fp.npy is missing` | The `casmi-train-cft-hor` download is incomplete: download and unzip it again. |
| PC gets very slow / swaps | Run `run_train_cft.bat --mmap` from a Command Prompt in the folder. |
| `non-finite loss ... batch skipped` | A few lines are harmless. On every line: stop and tell Shishir. |
| Anything else | Run it once more. If it fails the same way, send Shishir: `out_cft\train_log.txt`, a screenshot of the last 30 lines, and which step you were on. |


**Extra step for this job:** copy all files from the zip `akriti_hor_b.zip` into `C:\casmi-hor` (after step 2), replacing any with the same name.
