# RUNSHEET - evaluate fingerprint networks on the bench

This job scores a pair of trained fingerprint networks (for example the `fp_single_ho1.pt` / `fp_merged_ho1.pt`
you trained) on the team's test bench and sends the scores back. Nothing here needs programming.

- **Your attention:** about 20 minutes for the one-time setup. After that, each evaluation is one double-click.
- **Running time:** about 1.5-2.5 hours per evaluation, mostly spectral search on the processor (the graphics
  card makes little difference here).
- **Disk:** about **12 GB free** (2 GB package, 3 GB `train.parquet`, 6 GB Python/PyTorch, results).
- **Sleep:** set "When plugged in, put my device to sleep after" to **Never** (same as for training), and
  keep the laptop plugged in.

---

## One-time setup

1. **Get the package** (a private Kaggle dataset, shared with you):
   - website: open https://www.kaggle.com/datasets/shishiradhikari11/casmi-bench-pkg, click **Download**, then
     unzip it into a **new** folder, for example `C:\casmi-bench`; or
   - command line:
     ```
     kaggle datasets download shishiradhikari11/casmi-bench-pkg -p C:\casmi-bench --unzip
     ```
   The folder must contain `setup_bench.bat`, `run_bench.bat` and the sub-folders `research`, `external` and
   `results`. If you see `.zip` files named `research.zip`, `external.zip` or `results.zip`, right-click each one,
   choose **Extract All...** and extract it **into the same folder**.
2. **Join the competition once** (needed to download its data file): open
   https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/rules and accept the rules. Skip
   this if you have already joined.
3. **Double-click `setup_bench.bat`.** It installs Python packages (about 3 GB; be patient) and downloads
   `train.parquet` (2.9 GB). At the end it must print **`SETUP OK`**.
   (If you already have `train.parquet` from the team, copy it into the folder before this step; the download is
   then skipped.)
4. **Double-click `smoke_bench.bat`** (about 5 minutes). It must end with **`RESULT: ALL OK`**. This compares 5
   molecules with the team's own run and proves that your setup gives the same answers.

## Each evaluation

1. **Double-click `run_bench.bat`.** It asks two questions:
   - the folder that holds the two `.pt` files, for example `C:\casmi-train-pkg\out\export` (you can paste it
     with Explorer's "Copy as path"; the quotes are fine);
   - a short name for this run, without spaces, for example `ho1`.

   Leave it running. Lines like `channels 50/250` show progress (S1 and S2 have 250 molecules, S3 has 300). It ends with a small table,
   **`RESULT: ALL OK`** and **`FINISHED`**.
2. **Double-click `send_bench_results.bat`**, type the same run name and your Kaggle username. It uploads
   `results\bench\<name>` (about 100 MB) as a private dataset. Then open the page it prints, go to
   **Settings -> Sharing** and add **shishiradhikari11**.
3. Paste the text of `results\bench\<name>\bench_summary.txt` into the team chat.

### If it was interrupted (power cut, closed window, sleep)
Double-click **`run_bench.bat` again** and give the **same folder and the same name**. Finished parts (S1, S2, S3)
are kept; the unfinished part starts again from its beginning. Do not delete the `results` folder.

## What to do if ...
| Problem | Do this |
|---|---|
| "Python was not found" | Install Python 3.12 from https://www.python.org/downloads/windows/ and tick "Add python.exe to PATH". Then run `setup_bench.bat` again. |
| `train.parquet could not be downloaded` | Do setup step 2 (accept the competition rules), then run `setup_bench.bat` again. Or copy `train.parquet` from the team into the folder. |
| `SETUP NOT OK` with "train.parquet has ... bytes" | The download was incomplete. Delete `train.parquet` and run `setup_bench.bat` again. |
| `SETUP NOT OK` with "missing research/..." or "missing results/..." | The package was not unpacked completely. Redo setup step 1, including the **Extract All** part. |
| `SETUP NOT OK` with "rdkit is ..." or "sklearn is ..." | Delete the `venv` folder and run `setup_bench.bat` again. |
| `torch.cuda.is_available(): False` | That is fine for this job: the bench also runs on the processor. |
| `MemoryError` or the PC becomes unusably slow | Close other programs. Then open a Command Prompt in the folder and run `run_bench.bat "<folder>" <name> 1` (the `1` = use one worker: slower, less memory). Use the same folder and name as before; finished parts are kept. Tell the team you did this. |
| Smoke test says `RESULT: NOT OK` | Do not run the full bench. Report it (see below). |
| Anything else | Run it once more. If it fails the same way, report it. |

### What to report if it fails
Send:
- a screenshot or a copy of the last 30 lines in the window;
- the output of `setup_bench.bat` (the CHECK block);
- which step of this sheet you were on.

**Do not** change files in `research`, `external` or `results`: the comparison with the team's runs depends on
them being identical.
