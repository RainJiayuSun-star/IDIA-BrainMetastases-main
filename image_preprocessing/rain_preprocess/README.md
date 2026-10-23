# IDiA Lab Unified Preprocessing Pipeline (rain_preprocess)
==========================================================

Welcome to the **Unified Preprocessing Pipeline**! This folder contains a highly modular, clean, and reusable Python framework designed to run standard clinical neuroimaging operations on diverse datasets (such as **Yale/Brain-Mets-Lung** and **UCSF-BSMR**) from a single execution interface.

Instead of running separate standalone scripts for co-registration, skull-stripping, N4 bias correction, warping, and normalization, you can now define your entire preprocessing sequence inside a simple **YAML configuration file** or run targeted processing using dedicated step scripts.

---

## 📂 Folder Layout
*   `rain_preprocess.py`: The core pipeline orchestrator containing all processing algorithm wrappers (Steps 1, 2, 3, 5, 6).
*   `rain_preprocess_step3_bc.py`: Dedicated Step 3 N4 Bias Correction script featuring automatic Otsu thresholding fallback (for datasets without brain masks), PDF/Excel error logging, and automatic resume/skip tracking.
*   `rain_preprocess_step3_bc_multiworker.py`: Configurable CPU N4 with one image per task, configurable ITK threads, validated resume, and central bookkeeping.
*   `config_step3.example.yaml`: Dedicated multiworker N4 configuration; copy it and change paths for your dataset.
*   `config.yaml`: The orchestrator configuration mapping directories and step lists (a different schema from the dedicated N4 configuration).
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

### 1. Multi-process CPU N4
The dedicated multiworker N4 script runs one image per task. `num_workers` controls concurrent images, and `itk_threads_per_worker` controls the ITK thread limit inside each worker. Speed depends on image dimensions, memory bandwidth, storage, and available CPUs; linear scaling is not guaranteed. See the dedicated Step 3 instructions below for configuration and measured results.

The remaining synchronization and subset options in this section describe the unified orchestrator. The dedicated multiworker script uses its own configuration described below.

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

### 5. Run configurable multiworker CPU N4 (`rain_preprocess_step3_bc_multiworker.py`)

This script uses genuine SimpleITK N4, with one selected MRI image per task. Different images from the same patient may run on different workers. It does not use the GPU, even when the other pipeline steps do.

Install the required packages in your chosen Python environment:

```bash
python -m pip install SimpleITK numpy PyYAML
```

CSV reports are always written. Optional `pandas` plus `openpyxl` enable Excel export; `fpdf` enables PDF export. Missing optional exporters do not invalidate image processing.

Copy `config_step3.example.yaml` to your own configuration and edit the paths. This minimal example selects T1c and FLAIR and copies the segmentation unchanged:

```yaml
input_dir: '/data/dataset'
output_dir: null
case_list:
  path: null
  id_column: 'case_id'
images:
  t1c:
    filename: '{case_id}-t1c.nii.gz'
  flair:
    filename: '{case_id}-t2f.nii.gz'
brain_mask_filename: null
companions:
  mode: 'listed'
  filenames:
    - '{case_id}-seg.nii.gz'
num_workers: 4
itk_threads_per_worker: 1
max_retries: 3
retry_delay: 2.0
n4_params:
  shrink_factor: 2
  num_fitting_levels: 4
  num_iters: 50
  convergence_thresh: 0.000001
```

Preview before processing:

```bash
python rain_preprocess_step3_bc_multiworker.py --config my_step3.yaml --dry-run
python rain_preprocess_step3_bc_multiworker.py --config my_step3.yaml
```

Use `python3` where that is your interpreter command on Linux/macOS. YAML paths may use forward slashes on Windows. Relative configuration paths resolve against the YAML directory, and `~` expands to the home directory. Explicit CLI options override YAML, then script defaults apply:

```bash
python rain_preprocess_step3_bc_multiworker.py --config my_step3.yaml -i /data/input -o /data/corrected --num_workers 24 --itk_threads_per_worker 4
```

**Selection and output**

- `input_dir` directly contains visible case/scan folders. `{case_id}` is the exact folder name. Templates are exact filenames, not wildcards; fixed names and `.nii` inputs are supported too. Corrected outputs use `_BC.nii.gz`.
- Without a case list, all immediate visible folders are considered. TXT lists contain one exact folder ID per line, ignoring blanks and `#` comments. CSV lists use `case_list.id_column`. IDs retain leading zeros and are deduplicated; missing folders are reported. List entries must be folder IDs, not paths.
- Every configured image and explicitly configured mask is required; missing one skips the entire case. Each image may set `mask_filename` to override `brain_mask_filename`. Null means Otsu estimation for that image. A supplied brain mask must match the image geometry; a tumor segmentation is not a whole-brain mask.
- With `output_dir` omitted, null, or equal to `input_dir`, corrected images go beside originals. Originals are never overwritten. A separate output root preserves `<output_dir>/<case_id>/`. Other nested input/output root arrangements are rejected.
- Generated outputs are excluded from input discovery, preventing `_BC_BC` outputs on repeat runs. Dry runs report selections, missing files, destinations, companion copying, and CPU budgets without creating files or running N4.

**Companions and segmentation**

Companions are copied unchanged only for a separate destination. `none` copies nothing; `listed` copies exact `companions.filenames` templates; `all_unselected` copies other immediate original files, excluding generated outputs and processing artifacts. Subdirectories are not copied. Missing listed companions are logged without making the case ineligible. Companions are synchronized once all selected images for a patient finish, including validated resume skips, with a final synchronization at successful completion. In-place mode copies nothing.

Segmentation label maps must **not** receive N4 correction: their voxel values encode classes, not MRI signal intensities. Keep them out of `images` and include them as companions. ADC and other quantitative derived maps should also be preserved unchanged unless a separately validated protocol requires otherwise. The script follows your explicit image selection; it cannot infer the clinical meaning of every custom filename.

**Workers, threads, and measured performance**

The default is up to four workers (`min(4, os.cpu_count())`) and one ITK thread per worker. Positive integer `itk_threads_per_worker` values set the global ITK default in each worker before filters are constructed. Requested and effective counts and the ITK threader are recorded. The requested computation-thread budget is `num_workers * itk_threads_per_worker`; it is not guaranteed simultaneous utilization or a CPU reservation. Requests above process-visible logical CPUs warn and continue. Windows supports at most 61 worker processes with this pool implementation. The script does not automatically tune, pin CPUs, or configure NUMA placement.

One local Windows trial per setting on BMETS0048_01 T1c and FLAIR, using two workers and the same N4 parameters, measured:

| ITK threads per worker | Total wall time | Speedup versus one thread |
| --- | ---: | ---: |
| 1 | 116 seconds | 1.00× |
| 2 | 66 seconds | 1.76× |
| 4 | 50 seconds | 2.32× |

Changing thread count changes execution parallelism and can slightly change floating-point results. The largest tested foreground relative L2 difference from the one-thread output was approximately 0.0024%; this is numerical consistency evidence, not downstream clinical or segmentation validation. These exploratory local measurements do not establish an optimum for the 192-logical-CPU lab server. Benchmark worker/thread combinations within the CPU and RAM allocation of your job. Linux/macOS execution and remote-server performance remain unverified.

**Central bookkeeping, resume, and migration**

Patient folders contain corrected images and selected companions. Persistent per-output locks and completion records live under `<effective_output_root>/.n4_logs/outputs/`, keyed by SHA-256 of the canonical output path. Scan CSV reports, optional Excel/PDF reports, selection reports, and unfinished-task reports also live under `.n4_logs`.

The OS owns each active lock; the persistent lock file is its stable anchor. Do not delete central locks or `.n4_logs` while runs may be active. All independent runs targeting the same outputs must use this updated locking layout, on a filesystem with working advisory locks across clients. Separate output trees are independent jobs.

Completion records retain source/mask identities, N4 parameters, requested thread counts, effective execution settings, and output hashes. Resume validates the signature, full image decoding, geometry, finite values, and hash before skipping. Changing requested threads triggers recomputation. Signatures predating the thread setting are treated as one-thread results. Outputs without valid completion records are recomputed; moving an output tree may also require recomputation because provenance uses canonical paths.

**Stop every older script version targeting the output tree before the first updated run.** Old and new locking layouts must not run concurrently. Existing adjacent `<output>.complete.json` records remain readable. Only after all selected images verify and companion copying succeeds does the script migrate recognized legacy records centrally and remove their exact adjacent sidecars and retired one-byte lock anchors, including companion locks. It checks for active legacy lock holders, but cannot detect older jobs that have not acquired a lock yet. Unrecognized sidecars are preserved and reported as errors. The update does not retroactively sweep datasets: migration is limited to outputs and companions selected in a successful run.

Failure or interruption retains resume bookkeeping. New central locks persist after success; patient folders no longer need those files. No broad `.json` deletion occurs: source metadata, copied JSON companions, unrelated files, and central reports remain. Exit status is nonzero for failed/unfinished tasks or zero eligible cases; policy-driven skips are reported separately. Keep inputs immutable during processing.

Legacy presets and direct paths remain supported:

```bash
python rain_preprocess_step3_bc_multiworker.py -d 1 -w 4
python rain_preprocess_step3_bc_multiworker.py -i /data/input -o /data/corrected --sequences t1c t2f -w 4
```

Explicit `-i` without `-o` saves beside originals. `--config` cannot be mixed with dataset presets or legacy `--sequences`. N4 defaults remain shrink factor 2, four fitting levels, 50 iterations per level, and convergence threshold 1e-6; the correction remains `image / exp(full_resolution_log_bias_field)`.

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

The bullets below describe the unified orchestrator (`rain_preprocess.py`). The dedicated multiworker N4 script writes corrected images directly into case folders and bookkeeping under `.n4_logs`, as described above; its YAML mode requires PyYAML and uses a safe loader.
*   **Final Outputs**: Are mirrored perfectly by Patient ID directly under `output_dir/`.
*   **Intermediates**: To keep your final directory pristine, intermediate outputs of each step are stored separately under `output_dir/intermediates/step_X/` (e.g., `output_dir/intermediates/step_3/YG_0AXGKD8AFJGS/`).
*   **Zero-Dependency Fallback**: In case your active Python environment lacks PyYAML, the script includes a custom native parser fallback to parse the `config.yaml` seamlessly without throwing import crashes.
