# IDiA Lab Brain Metastases Image Preprocessing & Structuring Framework for Model Training

This directory contains the complete neuroimaging preprocessing, alignment, and dataset structuring framework developed by the IDiA Lab for processing multi-contrast patient brain MRI scans (T1CE, FLAIR, T1, T2) and tumor segmentation maps.

---

## 1. Directory Structure

```text
image_preprocessing/
├── rain_preprocess/                  # Unified Preprocessing Pipeline (Modern Orchestrator)
│   ├── rain_preprocess.py            # Core pipeline orchestrator (Steps 1, 2, 3, 5, 6)
│   ├── rain_preprocess_step3_bc.py   # Dedicated Step 3 N4 Bias Correction (Single-worker)
│   ├── rain_preprocess_step3_bc_multiworker.py # Multi-worker server edition (Parallel scaling)
│   ├── config.yaml                   # Configuration file mapping steps and inputs
│   └── config_brain_lung.yaml        # Pre-configured YAML template for Brain-Mets-Lung dataset
├── rearrangeStructure/               # Dataset Structure Rearranger Module
│   ├── rearrange_dataset.py          # Formats raw datasets into nnUNet or patient-folder layout
│   └── README.md                     # Documentation for dataset conversion
├── align/                            # Step 1: Alignment & Co-registration (Legacy & Utilities)
│   └── logging_COREG_RainScript.py   # Header resampling & mutual-information coregistration
├── Jared/                            # Steps 2, 3, 6: Legacy Preprocessing Scripts
│   ├── logging_jbHDBET_BrambleScript.py            # HD-BET skull stripping
│   ├── logging_jbBiasCorrection_BrambleScript_V2.py # N4 bias field correction
│   └── logging_jbIntensityNormalization_BrambleScript_V2.py # Intensity normalization
├── labeling/                         # Relabeling and label-map standardization utilities
├── templates/                        # Spatial registration target templates (MNI152)
├── Dockerfile.imgpreprocess          # Isolated Docker environment definition
└── Preprocessing Pipeline-May 2023 1.pdf # Official IDiA Lab guideline document
```

---

## 2. Guideline Step to Codebase Module Mapping

Based on the **IDiA Preprocessing Pipeline Guideline**, here is how each processing step maps to the codebase:

| Guideline Step | Description | Recommended Modern Script | Legacy / Alternative Script |
| :--- | :--- | :--- | :--- |
| **Step 1** | **Co-registration & Modal Alignment** | [rain_preprocess.py](file:///Users/rains/rainworkspace/2-IDIALab/IDIA-BrainMetastases-main/image_preprocessing/rain_preprocess/rain_preprocess.py) (Step 1) | [logging_COREG_RainScript.py](file:///Users/rains/rainworkspace/2-IDIALab/IDIA-BrainMetastases-main/image_preprocessing/align/logging_COREG_RainScript.py) |
| **Step 2** | **Skull Stripping / Brain Extraction** | [rain_preprocess.py](file:///Users/rains/rainworkspace/2-IDIALab/IDIA-BrainMetastases-main/image_preprocessing/rain_preprocess/rain_preprocess.py) (Step 2 HD-BET) | [logging_jbHDBET_BrambleScript.py](file:///Users/rains/rainworkspace/2-IDIALab/IDIA-BrainMetastases-main/image_preprocessing/Jared/logging_jbHDBET_BrambleScript.py) |
| **Step 3** | **N4 Bias Field Correction** | [rain_preprocess_step3_bc_multiworker.py](file:///Users/rains/rainworkspace/2-IDIALab/IDIA-BrainMetastases-main/image_preprocessing/rain_preprocess/rain_preprocess_step3_bc_multiworker.py) | [logging_jbBiasCorrection_BrambleScript_V2.py](file:///Users/rains/rainworkspace/2-IDIALab/IDIA-BrainMetastases-main/image_preprocessing/Jared/logging_jbBiasCorrection_BrambleScript_V2.py) |
| **Step 4** | **Tumor Segmentation Mask** | Ground Truth Masks (`_seg`) or **nnU-Net Model** | `nnUNet_raw` / Deep Learning Models |
| **Step 5** | **ANTs Diffeomorphic MNI Warp** | [rain_preprocess.py](file:///Users/rains/rainworkspace/2-IDIALab/IDIA-BrainMetastases-main/image_preprocessing/rain_preprocess/rain_preprocess.py) (Step 5) | ANTs CLI / ANTsPy Workflows |
| **Step 6** | **Intensity Normalization** | [rain_preprocess.py](file:///Users/rains/rainworkspace/2-IDIALab/IDIA-BrainMetastases-main/image_preprocessing/rain_preprocess/rain_preprocess.py) (Step 6) | [logging_jbIntensityNormalization_BrambleScript_V2.py](file:///Users/rains/rainworkspace/2-IDIALab/IDIA-BrainMetastases-main/image_preprocessing/Jared/logging_jbIntensityNormalization_BrambleScript_V2.py) |

---

## 3. Preprocessing Execution Workflows

### Workflow A: Unified Multimodal Pipeline (`rain_preprocess/rain_preprocess.py`)

Run the complete sequence (Co-registration $\rightarrow$ Skull Stripping $\rightarrow$ N4 Bias Correction $\rightarrow$ MNI Warp $\rightarrow$ Intensity Normalization) from a single execution interface:

```bash
cd image_preprocessing/rain_preprocess

# Run using custom configuration YAML file
python rain_preprocess.py -c config.yaml
```

### Workflow B: High-Speed Multi-Worker Server N4 Bias Correction (`rain_preprocess_step3_bc_multiworker.py`)

Optimized for processing large-scale clinical cohorts on high-core server infrastructure (e.g. 32 to 96 CPU cores):

```bash
cd image_preprocessing/rain_preprocess

# Run Dataset 1 (Brain-Mets-Lung) with 32 parallel worker processes
python rain_preprocess_step3_bc_multiworker.py -d 1 -w 32

# Run Dataset 2 (BraTS2025/2026 MET) with 64 parallel worker processes
python rain_preprocess_step3_bc_multiworker.py -d 2 -w 64
```

**Key Features**:
* **Parallel Scaling & Thread Isolation**: Spans worker processes (`-w 32/64`) while locking SimpleITK to single-threaded mode per worker to avoid CPU thrashing across NUMA sockets.
* **Atomic Scheduling & Lock-Free Multi-Process Execution**: Employs per-patient directory file locks (`.processing.lock`) for safe parallel worker execution.
* **Automatic Rescheduling & Verification**: Performs post-pass verification scans and automatically reschedules incomplete cases.
* **Centralized Reporting**: Aggregates process status into executive PDF and Excel summary logs.

---

## 4. Dataset Structuring & Conversion (`rearrangeStructure/`)

Before training or inference, raw clinical image directories can be converted into standard layout structures using [rearrange_dataset.py](file:///Users/rains/rainworkspace/2-IDIALab/IDIA-BrainMetastases-main/image_preprocessing/rearrangeStructure/rearrange_dataset.py):

### Target 1: nnU-Net Raw Dataset Format (Training)
```bash
cd image_preprocessing/rearrangeStructure

python rearrange_dataset.py \
  -i /path/to/raw_dataset \
  -o /path/to/nnUNet_raw \
  --dataset-id 101 \
  --dataset-name BraTS2026_MET \
  --file-action copy
```

### Target 2: Patient Directory Format (Inference / Preprocessing)
```bash
python rearrange_dataset.py \
  -i /path/to/raw_dataset \
  -o /path/to/preprocessed_input \
  --target-format patient_folders \
  --file-action copy
```

For full options and dry-run usage, refer to [rearrangeStructure/README.md](file:///Users/rains/rainworkspace/2-IDIALab/IDIA-BrainMetastases-main/image_preprocessing/rearrangeStructure/README.md).

---

## 5. Docker Environment & Deployment

To execute preprocessing inside an isolated environment on virtual machines or HPC servers, use [Dockerfile.imgpreprocess](file:///Users/rains/rainworkspace/2-IDIALab/IDIA-BrainMetastases-main/image_preprocessing/Dockerfile.imgpreprocess):

### 1. Build the Docker Image
```bash
cd image_preprocessing
docker build -t imgpreprocess -f Dockerfile.imgpreprocess .
```

### 2. Run Container with Volume Mounting
```bash
docker run -it -v /path/to/your/workspace:/app imgpreprocess
```

### 3. Execute Pipelines Inside Docker
```bash
python /app/image_preprocessing/rain_preprocess/rain_preprocess.py -c /app/config.yaml
```
