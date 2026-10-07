# Akriti: ensemble training on your laptop (about 35–45 hours, mostly unattended)

Hi Akriti! This trains **4 more fingerprint networks** with the same data you already used for the "ho1" run in
`C:\casmi-train-pkg`. We'll combine them into an ensemble, which should make our answers more accurate.
**Nothing gets submitted to the competition.**

**Start only after the CFT-ho2 run has finished** (one GPU job at a time).

**Time:** about 10 minutes of your attention. Then the laptop works alone for about **35–45 hours**. You can stop
it whenever you need the laptop and start it again later; it continues where it stopped.

## Before you start
- Keep the laptop **plugged in**. Sleep = **Never** (Settings → System → Power → Screen and sleep).
- About **3 GB free** on `C:`.
- Close games and other GPU programs.

## Steps
1. Shishir sends you 4 files: `train_fp.py`, `run_ensemble.bat`, `send_ensemble.bat`, `AKRITI_ENSEMBLE.md`.
   Copy them **into `C:\casmi-train-pkg`** (the folder you used for the first ho1 training, the one with the `venv`
   folder in it). When Windows asks, choose **"Replace the file in the destination"** for `train_fp.py`.
2. Double-click **`run_ensemble.bat`**.
   - It runs **JOB 1 of 4**, then 2, 3 and 4 automatically. Each of jobs 1–3 takes about 8 hours; job 4 is a
     bigger network and takes about 12–20 hours.
   - After about 15 minutes, send Shishir one line from the window (the one with `step/s`).
3. When the window says **`ALL 4 JOBS FINISHED`**, double-click **`send_ensemble.bat`** and type `akritirijal04`.
   Then open the page it prints → **Settings → Sharing** → add **shishiradhikari11**. Send Shishir the 4
   `*_check_summary.txt` texts from the `ens_upload` folder.

**Need the laptop in the middle?** Close the window. Later, double-click `run_ensemble.bat` again: finished jobs
are skipped and the current one continues from its last checkpoint (at most about 10 minutes are lost).

## If something goes wrong
| You see | Do this |
|---|---|
| `PROBLEM: this folder has no venv` | The files are in the wrong folder. Move them into `C:\casmi-train-pkg`. |
| `OUT OF GPU MEMORY` | Close other programs (browser with video, games), then double-click `run_ensemble.bat` again. |
| `NOT FINISHED` or `Stopped` | Double-click `run_ensemble.bat` again. |
| The same error twice | Send Shishir a photo of the last 30 lines. |
| `send_ensemble.bat` says the dataset already exists | Run the `kaggle datasets version` line it prints. |
| Some `check_summary` doesn't say `RESULT: ALL OK` | Send that text to Shishir anyway. |

Thank you! 🙏
