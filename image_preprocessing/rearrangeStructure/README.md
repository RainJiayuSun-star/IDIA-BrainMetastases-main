# Rearrange Data Folder Structure

nnUNet training and inference require specific dataset formats. This module provides a unified CLI script (`rearrange_dataset.py`) to convert raw clinical imaging datasets into desired target folder structures.

---

## 1. Supported Target Formats

### Target A: nnUNet Raw Dataset Format (Training)
Converts raw datasets into the official nnUNet v2 raw format required for training and evaluation.
- Documentation: [dataset_format.md](file:///Users/rains/rainworkspace/2-IDIALab/IDIA-BrainMetastases-main/train/nnUNet_rain/documentation/dataset_format.md)
- Output structure:
```text
nnUNet_raw/DatasetXXX_Name/
├── dataset.json
├── imagesTr/
│   ├── CASE_0001_0000.nii.gz  (Modality 0: T1CE)
│   ├── CASE_0001_0001.nii.gz  (Modality 1: FLAIR)
│   └── ...
└── labelsTr/
    ├── CASE_0001.nii.gz       (Segmentation map)
    └── ...
```

### Target B: Unified Preprocessing Format (Inference)
Converts raw datasets into patient-based directories for the IDiA Lab unified preprocessing pipeline (`rain_preprocess.py`).
- Documentation: [rain_preprocess/README.md](file:///Users/rains/rainworkspace/2-IDIALab/IDIA-BrainMetastases-main/image_preprocessing/rain_preprocess/README.md)
- Output structure:
```text
output_dir/
├── PATIENT_001/
│   ├── t1ce.nii.gz
│   ├── flair.nii.gz
│   └── seg.nii.gz
└── PATIENT_002/
    └── ...
```

---

## 2. CLI Usage & Command Examples

The script `rearrange_dataset.py` automatically detects dataset structures (BraTS or Yale Brain Mets) and arranges files into the designated format.

### Command Options
- `-i, --input-dir`: Path to raw input dataset folder.
- `-o, --output-dir`: Destination directory.
- `-t, --dataset-type`: `auto` (default), `brats`, or `yale`.
- `-f, --target-format`: `nnunet` (default) or `patient_folders`.
- `--dataset-id`: 3-digit dataset integer ID for nnUNet (e.g., `101`).
- `--dataset-name`: Name string for nnUNet dataset (e.g., `BraTS2026_MET`).
- `--file-action`: `copy` (default), `symlink` (saves space), or `move`.
- `--dry-run`: Preview file mappings without modifying filesystem.

---

### Example 1: Convert BraTS2026 MET Dataset to nnUNet Training Format
```bash
python rearrange_dataset.py \
  -i /Users/rains/rainworkspace/2-IDIALab/Datasets/BraTS2026/MICCAI-LH-BraTS2025-MET-Challenge-Training \
  -o /Users/rains/rainworkspace/2-IDIALab/IDIA-BrainMetastases-main/train/nnUNet_rain/nnUNet_raw \
  --dataset-id 101 \
  --dataset-name BraTS2026_MET \
  --file-action copy
```

### Example 2: Convert Yale Brain Mets Dataset to nnUNet Format (using symlinks to save space)
```bash
python rearrange_dataset.py \
  -i /Users/rains/rainworkspace/2-IDIALab/Datasets/PKG_Brain-Mets-Lung-MRI-Path-Segs_radiology_images \
  -o /Users/rains/rainworkspace/2-IDIALab/IDIA-BrainMetastases-main/train/nnUNet_rain/nnUNet_raw \
  --dataset-id 102 \
  --dataset-name Yale_Brain_Mets \
  --file-action symlink
```

### Example 3: Convert BraTS Dataset to Patient Directory Format for `rain_preprocess.py`
```bash
python rearrange_dataset.py \
  -i /Users/rains/rainworkspace/2-IDIALab/Datasets/BraTS2026/MICCAI-LH-BraTS2025-MET-Challenge-Training \
  -o /Users/rains/rainworkspace/2-IDIALab/Datasets/BraTS2026_Preprocess_Input \
  --target-format patient_folders \
  --file-action copy
```

### Example 4: Dry-Run Preview
To inspect case discovery and file mappings before execution:
```bash
python rearrange_dataset.py \
  -i /Users/rains/rainworkspace/2-IDIALab/Datasets/BraTS2026/MICCAI-LH-BraTS2025-MET-Challenge-Training \
  -o /tmp/test \
  --dry-run
```