# Step-by-Step Guide for Binaya (RTX 3050 Laptop Training)

Hi Binaya! 👋
This guide will walk you through setting up and running a model training job on your laptop's **RTX 3050 GPU**. 

You do **not** need to write any code or use complex terminal commands. Everything is automated through 4 simple double-click files.

---

### 📋 Before You Begin
1. **Keep your laptop plugged in** to the charger at all times during training.
2. **Prevent your laptop from sleeping**:
   * Windows **Settings** $\to$ **System** $\to$ **Power & battery** $\to$ **Screen and sleep**.
   * Set *"When plugged in, put my device to sleep after"* to **Never**.
3. Make sure you have at least **5 GB free space** on your drive.
4. Close heavy 3D games or video editing software while training runs.

---

### 🚀 Step 1: Install Python (If you don't have it yet)
If you already have Python installed, skip to Step 2.
1. Download Python 3.12 (64-bit) from the official site:
   👉 **[https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)**
2. Run the installer.
3. ⚠️ **VERY IMPORTANT**: On the very first screen of the installer, check the box at the bottom:
   * **`[X] Add python.exe to PATH`**
4. Click **Install Now**.

---

### 📂 Step 2: Extract the Training Folder
1. Shishir will send you a zip folder named `casmi-binaya.zip`.
2. Extract the zip file directly to `C:\casmi-train` (or anywhere on your main drive).
3. Inside the folder, you will see files named:
   * `1_setup.bat`
   * `2_test_gpu.bat`
   * `3_run_training.bat`
   * `4_send_results.bat`
   * and the data files (`.npy`, `.py`, etc.)

---

### ⚙️ Step 3: Run the One-Time Setup
1. Double-click **`1_setup.bat`**.
2. A black terminal window will open and automatically install PyTorch with NVIDIA CUDA support for your RTX 3050.
3. This download takes about 3–5 minutes depending on your internet speed.
4. When it finishes and says *"Setup completed!"*, press any key to close the window.

---

### 🎮 Step 4: Verify Your RTX 3050 GPU
1. Double-click **`2_test_gpu.bat`**.
2. It will display your GPU name (e.g. `NVIDIA GeForce RTX 3050 Laptop GPU`) and run a quick 5-second test.
3. If it says:
   `SUCCESS: Your GPU is working perfectly!`
   You are completely ready to train!

---

### 🏃 Step 5: Start Model Training
1. Double-click **`3_run_training.bat`**.
2. The training will start immediately.
   * You will see progress lines showing the current step and speed (e.g., `step/s`).
   * Total runtime is about **8 to 10 hours** on your RTX 3050.
3. **Can I pause or stop it?**
   * **Yes!** You can close the window anytime you need your laptop.
   * When you are ready to continue, simply double-click **`3_run_training.bat`** again. It will automatically resume right where it left off!

---

### 📦 Step 6: Sending the Results
1. When training finishes, the window will say:
   `TRAINING COMPLETE! Now double-click 4_send_results.bat`
2. Double-click **`4_send_results.bat`**.
3. It will gather your trained model files into a folder named **`binaya_results`**.
4. Zip the `binaya_results` folder and send it to Shishir via Google Drive, Telegram, or WhatsApp!

---

### ❓ What if an error happens?
* **"Out of GPU Memory"**: Close your browser or any other app using graphics, and run `3_run_training.bat` again.
* **Any other error**: Take a screenshot or photo of the window and send it to Shishir. He will fix it right away!

Thank you for helping the team! 🚀
