# Binaya: CFT "full_b" training on your laptop (RTX 3050)

Hi Binaya! Thanks for the last run - it worked. This is the next one: a bigger network on the full data. About
**15-25 hours** of laptop time; you can stop and restart any time. **Nothing gets submitted.**

**Before you start:** keep the laptop plugged in, sleep set to Never, close games. Need about **15 GB free** on C:.

1. Make an empty folder **C:\casmi-full** and unzip **binaya_full_b.zip** into it.
2. Double-click **download_full.bat**. It downloads about 3.6 GB from your own Kaggle account (the Kaggle login is the
   kaggle.json Shishir set up; if it says kaggle.json not found, tell Shishir). It must end with **ALL FILES DOWNLOADED**.
   If some files failed, double-click it again.
3. Double-click **setup_windows.bat**. It must end with **SETUP OK** and `torch.cuda.is_available(): True`.
4. Optional 5-minute test: **smoke_test_cft.bat** -> must end with **RESULT: ALL OK**.
5. Double-click **run_train_cft.bat**. After 15 minutes send Shishir one line with `step/s` and `vram`.
   If the window closes or the laptop restarts: double-click run_train_cft.bat again - it continues.
   If it says **OUT OF GPU MEMORY**: open cmd in the folder and run `run_train_cft.bat --bs 64`, and tell Shishir.
6. When it says **FINISHED**: double-click **check_result_cft.bat** -> must say **RESULT: ALL OK**.
7. Double-click **send_results_cft.bat**, type `binayaadhikari13`. Then send Shishir the text of
   `out_cft\export\check_summary_cft.txt`.

Thank you! 🙏
