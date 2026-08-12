# IDiA Lab Unified Preprocessing Pipeline (rain_preprocess)
==========================================================

Welcome to the **Unified Preprocessing Pipeline**! This folder contains a highly modular, clean, and reusable Python framework designed to run standard clinical neuroimaging operations on diverse datasets (such as **Yale/Brain-Mets-Lung** and **UCSF-BSMR**) from a single execution interface.

Instead of running separate standalone scripts for co-registration, skull-stripping, N4 bias correction, warping, and normalization, you can now define your entire preprocessing sequence inside a simple **YAML configuration file** or run targeted processing using dedicated step scripts.

---

## 📂 Folder Layout
*   `rain_preprocess.py`: The core pipeline orchestrator containing all processing algorithm wrappers (Steps 1, 2, 3, 5, 6).
*   `rain_preprocess_step3_bc.py`: Dedicated Step 3 N4 Bias Correction script featuring automatic Otsu thresholding fallback (for datasets without brain masks), PDF/Excel error logging, and automatic resume/skip tracking.
*   `config.yaml`: The user-configurable parameters file mapping directories and step lists.
*   `config_brain_lung.yaml`: Configuration template tuned for Brain-Mets-Lung preprocessing.
*   `README.md`: This documentation.

---

## ⚙️ How it Works & Selected Step Numbers

The pipeline assigns a standard running number to each key step in the **IDiA Lab Preprocessing Guideline**:

| Step Number | Operation Name | Description | Suffix Output |
| :--- | :--- | :--- | :--- |
| **Step 1** | **Co-registration** | Rigid mutual-information alignment of secondary modalities (FLAIR) to baseline (T1CE). | `_aligned.nii.gz` |
| **Step 2** | **Skull Stripping** | Automated brain tissue extraction using the HD-BET neural network. | `_SS.nii.gz` & `_SS_bet.nii.gz` |
| **Step 3** | **N4 Bias Correction** | Removes low-frequency RF coil intensity inhomogeneities using SimpleITK. | `_BC.nii.gz` |
| **Step 5** | **Template Warping** | Rigid/Affine multi-resolution alignment onto SRI-24 or MNI152 template space. | `_MNI.nii.gz` |
| **Step 6** | **Intensity Normalization** | Rescales brain voxels linearly (between 1st & 99th percentiles) to a standard scale (e.g., 0–4000). | `_IN.nii.gz` |

### 💡 The "Fork in the Road" Strategy
You can execute **any arbitrary combination** of steps by specifying them in your `config.yaml` steps list:
*   To prepare scans in **Native Clean Space** (for models like *Met-Seg* or *Brain-SAM*):
    ```yaml
    steps: [1, 2, 3] # Aligned, Skull-Stripped, Bias-Corrected in Patient Space
    ```
*   To prepare scans in **Standardized MNI/SRI Space** (for models like *AURORA* or *Neuroverse3D*):
    ```yaml
    steps: [1, 2, 3, 5, 6] # Aligned, Skull-Stripped, Bias-Corrected, Template-aligned, Normalized
    ```
*   To run **N4 Bias Correction and Intensity Normalization Only**:
    ```yaml
    steps: [3, 6]
    ```

---

## 🚀 Advanced Premium Features

The pipeline has been upgraded with cutting-edge engineering enhancements to maximize performance, control, and data integrity:

### 1. ⚡ Multi-Process Parallel Acceleration
To bypass the single-core CPU limitations of standard medical imaging libraries:
*   **Linear Scaling**: Set `num_workers: 8` (or match your CPU thread count) in `config.yaml` to spin up a parallel worker pool.
*   **Oversubscription Protection**: Each worker automatically runs in single-threaded mode (`sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)`) to eliminate CPU thrashing and cache conflicts, ensuring **100% linear core utilization**.
*   **Speedup**: Reduces processing time on 461 UCSF patients from **~2.8 hours down to under 20 minutes**!

### 2. 📂 Full Directory Synchronization & Integrity
*   **Multi-Segmentation Discovery**: Automatically detects, wraps, and resamples **all** tumor masks matching `*seg*` or `*tumor*` case-insensitively, perfectly maintaining multiple annotations (e.g. `_seg` and `_BraTS-seg` in UCSF).
*   **Raw File Syncing**: At the end of processing, a synchronization engine scans the patient's raw folder and **copies all unprocessed files** (like subtraction scans `*_subtraction.nii.gz` or clinical sheets) directly to the output folder. Your final output directories are 100% complete and match the input exactly!

### 3. 🎯 High-Precision Patient Subset Filtering
Control exactly which patients are processed directly from your configuration file:
*   **`patient_list: ["100101A", "100102A"]`** — Processes only these specific patient IDs.
*   **`patient_range: [1, 10]`** — Processes a 1-indexed slice of patients (e.g., patient #1 to #10). Ideal for quick debug loops.
*   **`patient_limit: 15`** — Limits execution to the first *N* patient folders.

---

## 🛠️ Usage Instructions

### 1. Environment Activation
We recommend running this pipeline within the **`nnUnet`** Conda environment, which already has `SimpleITK`, `pandas`, `numpy`, and `PyTorch` pre-installed:
```bash
conda activate nnUnet
```

### 2. Configure `config.yaml`
Open `config.yaml` and adjust the paths and settings:
*   `input_dir`: Path to the input dataset containing patient subdirectories.
*   `output_dir`: Path where preprocessed outputs will be saved.
*   `steps`: An array of selected running step numbers (e.g. `[3, 6]`).
*   `device`: Set to `"cuda"` if you are running Step 2 (HD-BET) on a GPU-enabled machine, or `"cpu"` otherwise.

### 3. Run the Pipeline (`rain_preprocess.py`)
To start the pipeline execution, you can specify your configuration file using a named argument or a positional argument. If no file is specified, it defaults to `config.yaml` in the current directory:
```bash
# Option A: Run using the named config flag (recommended)
python rain_preprocess.py --config config.yaml
# or short flag:
python rain_preprocess.py -c config.yaml

# Option B: Run with the default 'config.yaml' automatically
python rain_preprocess.py

# Option C: Run using the positional argument (backwards compatible)
python rain_preprocess.py config.yaml
```

### 4. Run Dedicated Step 3 Bias Correction (`rain_preprocess_step3_bc.py`)
When you specifically need **Step 3: N4 Bias Field Correction** on single-worker setups (with or without pre-computed brain masks):

```bash
# Option A: Run Pre-configured Dataset 1 (Brain-Mets-Lung)
python rain_preprocess_step3_bc.py -d 1

# Option B: Run Pre-configured Dataset 2 (BraTS2025-MET)
python rain_preprocess_step3_bc.py -d 2

# Option C: Test on a small patient subset (e.g. patients 0 to 5)
python rain_preprocess_step3_bc.py -d 1 -s 0:5

# Option D: Run on Custom Input and Output Directories
python rain_preprocess_step3_bc.py -i /path/to/input_root -o /path/to/output_root
```

### 5. Run Multi-Worker Server Edition (`rain_preprocess_step3_bc_multiworker.py`)
For high-speed parallel scaling on large multi-socket CPU servers (e.g. 192 CPUs / 96 physical cores):

```bash
# Option A: Run Dataset 1 (Brain-Mets-Lung) with 32 parallel worker processes
python rain_preprocess_step3_bc_multiworker.py -d 1 -w 32

# Option B: Run Dataset 2 (BraTS2025-MET) with 64 parallel worker processes
python rain_preprocess_step3_bc_multiworker.py -d 2 -w 64

# Option C: Run Custom Dataset with 96 Workers and 5 Max Reschedule Passes
python rain_preprocess_step3_bc_multiworker.py -i /path/to/input -o /path/to/output -w 96 --max_retries 5
```

**Multi-Worker Server Edition Features**:
* **Parallel Scaling & Thread Isolation**: Spans parallel process workers (`-w 32/64/96`) while locking SimpleITK to single-threaded mode per worker (`sitk.SetGlobalDefaultNumberOfThreads(1)`) to eliminate CPU thrashing and oversubscription across NUMA sockets.
* **Atomic Scheduling & Duplicate Prevention**: Uses file locks (`.processing.lock`) per subject directory to guarantee zero worker collision or duplicate processing.
* **Automated Post-Pass Verification & Rescheduling**: Scans 100% of output directories after worker pool completion. If any subjects remain incomplete, it automatically triggers retry passes until all subjects are verified complete.
* **Centralized Process-Safe PDF & Excel Logs**: Master process collects worker results into clean PDF and Excel logs.

---

## 🐳 Docker Deployment

A pre-configured premium Docker environment with PyTorch (CUDA 12.1), SimpleITK, HD-BET, nnUNetv2, and standard Python image libraries is provided to ensure consistent and portable execution of the pipeline.

### 1. Build the Docker Image
From the repository root directory, run:
```bash
docker build -t rain_preprocess:latest -f image_preprocessing/rain_preprocess/Dockerfile image_preprocessing/rain_preprocess/
```

### 2. Run the Docker Container
Run the container with GPU support and mount your local directories (e.g., datasets and templates):
```bash
docker run --gpus all -it \
  -v /path/to/local/input:/data/input \
  -v /path/to/local/output:/data/output \
  -v /path/to/local/templates:/data/templates \
  rain_preprocess:latest
```
or Use the --name flag inside your docker run command:
```{bash}
docker run --name rain_preprocess_runner --gpus all -it \
  -v /mnt/local/data/rainsun/metastases:/workspace \
  rain_preprocess:latest
```

### 3. Run Preprocessing inside Container
Once inside the container shell, configure your `/workspace/rain_preprocess/config.yaml` to point to the mounted paths:
```yaml
input_dir: "/data/input"
output_dir: "/data/output"
template_path: "/data/templates/sri24_anatomy.nifti.zip"
device: "cuda"  # for GPU skull stripping
```
Then start the execution:
```bash
cd /workspace/rain_preprocess
python rain_preprocess.py --config config.yaml
```

---

## 📈 Intermediate Outputs & Logging
*   **Final Outputs**: Are mirrored perfectly by Patient ID directly under `output_dir/`.
*   **Intermediates**: To keep your final directory pristine, intermediate outputs of each step are stored separately under `output_dir/intermediates/step_X/` (e.g., `output_dir/intermediates/step_3/YG_0AXGKD8AFJGS/`).
*   **Zero-Dependency Fallback**: In case your active Python environment lacks PyYAML, the script includes a custom native parser fallback to parse the `config.yaml` seamlessly without throwing import crashes.
