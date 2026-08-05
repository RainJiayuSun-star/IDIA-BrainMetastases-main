# Brain Metastases Segmentation: Master Experimentation & Benchmark Summary (June 30)

This report compiles and analyzes all training configurations and validation results completed so far on the **Dataset001 (UCSF Brain Mets)** cohort (Fold 0).

---

## 1. Experiment Overview & Setup Details

We have conducted five main experimental stages, transitioning from classical baseline U-Net architectures using standard class labels to customized region-based training strategies under varying epoch structures.

### Experiment 1: Standard Baseline (`nnUNetTrainer` + Plain UNet)
*   **Architecture:** Plain 3D U-Net (standard nnU-Net design).
*   **Trainer & Plans:** [nnUNetTrainer](file:///Users/rains/rainworkspace/2-IDIALab/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset/nnUNet_results/Dataset001_UCSFbrainmets/nnUNetTrainer__nnUNetPlans__3d_fullres) with standard `nnUNetPlans`.
*   **Loss Function:** Standard nnU-Net Loss (Deep Supervision wrapper of 1:1 Dice Loss + Cross-Entropy Loss).
*   **Optimizer:** Stochastic Gradient Descent (SGD) with Nesterov Momentum (0.99).
*   **Initial Learning Rate:** $10^{-2}$ (polynomial schedule decay).
*   **Weight Decay:** $3 \times 10^{-5}$.
*   **Regularization:** None (`dropout_op = null`).
*   **Batch Size:** 2 | **Patch Size:** $[80, 192, 160]$ (Median spacing: $[1.5, 0.86, 0.86]$).
*   **Labels:** Class-based mutually exclusive labels:
    *   Class 1: Edema
    *   Class 2: Necrosis / Non-enhancing tumor
    *   Class 3: Enhancing active tumor
*   **Training Length:** 1000 epochs (analyzed at Epoch 365).

### Experiment 2: Standard Residual Encoder (`nnUNetTrainer` + ResEnc UNet)
*   **Architecture:** Large Residual Encoder U-Net.
*   **Trainer & Plans:** [nnUNetTrainer](file:///Users/rains/rainworkspace/2-IDIALab/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset/nnUNet_results/Dataset001_UCSFbrainmets/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres) with standard `nnUNetResEncUNetLPlans`.
*   **Loss Function:** Standard nnU-Net Loss (Deep Supervision wrapper of Dice + Cross-Entropy).
*   **Optimizer:** SGD with Nesterov Momentum (0.99).
*   **Initial Learning Rate:** $10^{-2}$ (polynomial scheduler).
*   **Weight Decay:** $3 \times 10^{-5}$.
*   **Regularization:** None.
*   **Batch Size:** 5 | **Patch Size:** $[96, 224, 160]$.
*   **Labels:** Class-based mutually exclusive labels.
*   **Training Length:** 1000 epochs (paused/stopped early at Epoch 776).

### Experiment 3: Custom Residual Encoder (`nnUNetTrainerMets` + ResEnc UNet)
*   **Architecture:** Large Residual Encoder U-Net.
*   **Trainer & Plans:** [nnUNetTrainerMets](file:///Users/rains/rainworkspace/2-IDIALab/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset/nnUNet_results/Dataset001_UCSFbrainmets/nnUNetTrainerMets__nnUNetResEncUNetLPlans_custom_0624__3d_fullres) with custom `nnUNetResEncUNetLPlans_custom_0624`.
*   **Loss Function:** Standard nnU-Net Loss (Deep Supervision wrapper of Dice + Cross-Entropy).
*   **Optimizer:** AdamW (`amsgrad = True`).
*   **Initial Learning Rate:** $3 \times 10^{-4}$ (polynomial schedule decay).
*   **Weight Decay:** $1 \times 10^{-4}$.
*   **Regularization:** 3D Spatial Dropout ($p = 0.2$ in encoder stages).
*   **Batch Size:** 4 | **Patch Size:** $[96, 224, 160]$.
*   **Labels:** Class-based mutually exclusive labels.
*   **Training Length:** 500 epochs (stopped early at Epoch 142 due to severe validation loss divergence and collapse).

### Experiment 4: Configurable Residual Encoder with Region-Based Training (`nnUNetTrainerMetsConfigurable` + ResEnc UNet)
*   **Architecture:** Large Residual Encoder U-Net.
*   **Trainer & Plans:** `nnUNetTrainerMetsConfigurable` with custom plans `nnUNetResEncUNetLPlans_custom_0624`.
*   **Loss Function:** Region-Based Loss Wrapper (`WeightedRegionLossWrapper` applying deep supervision over region-specific Dice + Binary Cross-Entropy losses).
*   **Optimizer:** AdamW.
*   **Batch Size:** 4 | **Patch Size:** $[160, 192, 160]$ (Median spacing: $[1.5, 0.86, 0.86]$).
*   **Training Length:** 300 epochs.
*   **Labels:** Hierarchical overlapping regions:
    *   Region 1: Whole Tumor (WT) = `label_1 + label_2 + label_3` (Edema + Necrosis + Enhancing)
    *   Region 2: Tumor Core (TC) = `label_2 + label_3` (Necrosis + Enhancing)
    *   Region 3: Enhancing Tumor (ET) = `label_3` (Enhancing active tumor)
*   **Sub-configurations (Runs 4.1 – 4.4):**
    *   [Run 4.1 (`exp_lr_1e-4_wd_5e-4`)](file:///Users/rains/rainworkspace/2-IDIALab/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset/nnUNet_results/Dataset001_UCSFbrainmets/nnUNetTrainerMetsConfigurable_exp_lr_1e-4_wd_5e-4__nnUNetResEncUNetLPlans_custom_0624__3d_fullres): LR = $1 \times 10^{-4}$ | WD = $5 \times 10^{-4}$ | Dropout = $0.3$ | Region Weights = `[1.0, 1.0, 2.0]`
    *   [Run 4.2 (`exp_lr_2e-4_dropout_0.4`)](file:///Users/rains/rainworkspace/2-IDIALab/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset/nnUNet_results/Dataset001_UCSFbrainmets/nnUNetTrainerMetsConfigurable_exp_lr_2e-4_dropout_0.4__nnUNetResEncUNetLPlans_custom_0624__3d_fullres): LR = $2 \times 10^{-4}$ | WD = $3 \times 10^{-4}$ | Dropout = $0.4$ | Region Weights = `[1.0, 1.0, 2.0]`
    *   [Run 4.3 (`exp_lr_2e-4_wd_3e-4`)](file:///Users/rains/rainworkspace/2-IDIALab/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset/nnUNet_results/Dataset001_UCSFbrainmets/nnUNetTrainerMetsConfigurable_exp_lr_2e-4_wd_3e-4__nnUNetResEncUNetLPlans_custom_0624__3d_fullres): LR = $2 \times 10^{-4}$ | WD = $3 \times 10^{-4}$ | Dropout = $0.3$ | Region Weights = `[1.0, 1.0, 2.0]`
    *   [Run 4.4 (`exp_lr_3e-4_wd_1e-4_dropout_0.2`)](file:///Users/rains/rainworkspace/2-IDIALab/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset/nnUNet_results/Dataset001_UCSFbrainmets/nnUNetTrainerMetsConfigurable_exp_lr_3e-4_wd_1e-4_dropout_0.2__nnUNetResEncUNetLPlans_custom_0624__3d_fullres): LR = $3 \times 10^{-4}$ | WD = $1 \times 10^{-4}$ | Dropout = $0.2$ | Region Weights = `[1.0, 1.0, 1.0]`

### Experiment 5 (Run 5.1): Extended Training Epochs (`nnUNetTrainerMetsConfigurable` + ResEnc UNet)
*   **Architecture:** Large Residual Encoder U-Net.
*   **Trainer & Plans:** [nnUNetTrainerMetsConfigurable](file:///Users/rains/rainworkspace/2-IDIALab/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset/nnUNet_results/Dataset001_UCSFbrainmets/nnUNetTrainerMetsConfigurable_run5_1__nnUNetResEncUNetLPlans_custom_0624__3d_fullres) using `nnUNetResEncUNetLPlans_custom_0624`.
*   **Loss Function:** Weighted Region-Based Loss Wrapper (`[1.0, 1.0, 2.0]` weighting).
*   **Optimizer:** AdamW.
*   **Initial Learning Rate:** $1 \times 10^{-4}$ | **Weight Decay:** $5 \times 10^{-4}$ | **Dropout:** $0.3$. (Hyperparameters identical to Run 4.1).
*   **Batch Size:** 4 | **Patch Size:** $[160, 192, 160]$.
*   **Labels:** Region-based (WT, TC, ET).
*   **Training Length:** **800 epochs** (completed).

---

## 2. Master Comparison Table

The table below outlines the results for all runs. 
*   **Pseudo-Dice** refers to the patch-level evaluation on validation patches sampled on the fly during training (as printed in the training logs).
*   **Final Validation Dice** is the full 3D resolution evaluation over the entire validation fold grid (recorded inside `summary.json`).

| Run ID / Folder | Optimizer | Initial LR | Weight Decay | Loss Function | Label Target | Patch Size | Dropout | Epochs | Region Weights | Final Train Loss | Final Val Loss | Latest Pseudo-Dice* | Val Mean Dice | Region/Class 1 (Edema / WT) | Region/Class 2 (Necrosis / TC) | Region/Class 3 (Enhancing / ET) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **Exp 1: Baseline** | SGD | $1 \times 10^{-2}$ | $3 \times 10^{-5}$ | Dice + CE | Class-Based | $[80, 192, 160]$ | None | 365 / 1000 | N/A | $\sim -0.3500$ | $\sim -0.2100$ | N/A | **$25.33\%$** | $40.13\%$ (Edema) | $20.32\%$ (Necrosis) | $15.53\%$ (Enhancing) |
| **Exp 2: SGD ResEnc** | SGD | $1 \times 10^{-2}$ | $3 \times 10^{-5}$ | Dice + CE | Class-Based | $[96, 224, 160]$ | None | 776 / 1000 | N/A | $-0.7082$ | $-0.4419$ | `[63.65, 25.61, 27.94]` | **N/A** (No Val) | N/A | N/A | N/A |
| **Exp 3: Custom ResEnc**| AdamW | $3 \times 10^{-4}$ | $1 \times 10^{-4}$ | Dice + CE | Class-Based | $[96, 224, 160]$ | Dropout (0.2) | 142 / 500 | N/A | $-0.6240$ | $-0.3335$ | `[68.34, 39.69, 28.82]` | **$25.38\%$** | $46.56\%$ (Edema) | $15.67\%$ (Necrosis) | $13.91\%$ (Enhancing) |
| **Run 4.1** | AdamW | $1 \times 10^{-4}$ | $5 \times 10^{-4}$ | Weighted Region | Region-Based | $[160, 192, 160]$ | Dropout (0.3) | 300 / 300 | `[1.0, 1.0, 2.0]` | $-0.4017$ | $-0.3188$ | `[56.58, 45.76, 52.70]` | **$39.17\%$** | $61.57\%$ (WT) | $24.52\%$ (TC) | $31.42\%$ (ET) |
| **Run 4.2** | AdamW | $2 \times 10^{-4}$ | $3 \times 10^{-4}$ | Weighted Region | Region-Based | $[160, 192, 160]$ | Dropout (0.4) | 300 / 300 | `[1.0, 1.0, 2.0]` | $-0.4002$ | $-0.3201$ | `[62.02, 40.04, 38.59]` | **$38.31\%$** | $59.81\%$ (WT) | $23.92\%$ (TC) | $31.19\%$ (ET) |
| **Run 4.3** | AdamW | $2 \times 10^{-4}$ | $3 \times 10^{-4}$ | Weighted Region | Region-Based | $[160, 192, 160]$ | Dropout (0.3) | 300 / 300 | `[1.0, 1.0, 2.0]` | $-0.4285$ | $-0.3653$ | `[64.50, 51.95, 55.17]` | **$39.12\%$** | $62.35\%$ (WT) | $25.04\%$ (TC) | $29.97\%$ (ET) |
| **Run 4.4** | AdamW | $3 \times 10^{-4}$ | $1 \times 10^{-4}$ | Weighted Region | Region-Based | $[160, 192, 160]$ | Dropout (0.2) | 300 / 300 | `[1.0, 1.0, 1.0]` | $-0.4905$ | $-0.3441$ | `[50.12, 26.89, 28.94]` | **$35.87\%$** | $61.59\%$ (WT) | $22.24\%$ (TC) | $23.80\%$ (ET) |
| **Run 5.1** | AdamW | $1 \times 10^{-4}$ | $5 \times 10^{-4}$ | Weighted Region | Region-Based | $[160, 192, 160]$ | Dropout (0.3) | 800 / 800 | `[1.0, 1.0, 2.0]` | $-0.4527$ | $-0.3243$ | `[70.32, 52.58, 56.93]` | **$35.15\%$** | $64.57\%$ (WT) | $19.46\%$ (TC) | $21.43\%$ (ET) |

*\*Note: For class-based runs (Exp 1-3), Pseudo-Dice is reported on mutually exclusive classes [Edema, Necrosis, Enhancing]. For region-based runs (Exp 4 & 5), it is reported on overlapping regions [WT, TC, ET].*

---

## 3. Training Log Links

Direct paths to the training logs for verification:
*   **Exp 2 Log:** [training_log_2026_6_23_04_27_31.txt](file:///Users/rains/rainworkspace/2-IDIALab/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset/nnUNet_results/Dataset001_UCSFbrainmets/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_0/training_log_2026_6_23_04_27_31.txt)
*   **Exp 3 Log:** [training_log_2026_6_24_17_20_00.txt](file:///Users/rains/rainworkspace/2-IDIALab/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset/nnUNet_results/Dataset001_UCSFbrainmets/nnUNetTrainerMets__nnUNetResEncUNetLPlans_custom_0624__3d_fullres/fold_0/training_log_2026_6_24_17_20_00.txt)
*   **Run 4.1 Log:** [training_log_2026_6_26_16_38_23.txt](file:///Users/rains/rainworkspace/2-IDIALab/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset/nnUNet_results/Dataset001_UCSFbrainmets/nnUNetTrainerMetsConfigurable_exp_lr_1e-4_wd_5e-4__nnUNetResEncUNetLPlans_custom_0624__3d_fullres/fold_0/training_log_2026_6_26_16_38_23.txt)
*   **Run 4.2 Log:** [training_log_2026_6_27_01_08_40.txt](file:///Users/rains/rainworkspace/2-IDIALab/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset/nnUNet_results/Dataset001_UCSFbrainmets/nnUNetTrainerMetsConfigurable_exp_lr_2e-4_dropout_0.4__nnUNetResEncUNetLPlans_custom_0624__3d_fullres/fold_0/training_log_2026_6_27_01_08_40.txt)
*   **Run 4.3 Log:** [training_log_2026_6_26_08_07_54.txt](file:///Users/rains/rainworkspace/2-IDIALab/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset/nnUNet_results/Dataset001_UCSFbrainmets/nnUNetTrainerMetsConfigurable_exp_lr_2e-4_wd_3e-4__nnUNetResEncUNetLPlans_custom_0624__3d_fullres/fold_0/training_log_2026_6_26_08_07_54.txt)
*   **Run 4.4 Log:** [training_log_2026_6_27_09_38_01.txt](file:///Users/rains/rainworkspace/2-IDIALab/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset/nnUNet_results/Dataset001_UCSFbrainmets/nnUNetTrainerMetsConfigurable_exp_lr_3e-4_wd_1e-4_dropout_0.2__nnUNetResEncUNetLPlans_custom_0624__3d_fullres/fold_0/training_log_2026_6_27_09_38_01.txt)
*   **Run 5.1 Log:** [training_log_2026_6_29_02_57_13.txt](file:///Users/rains/rainworkspace/2-IDIALab/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset/nnUNet_results/Dataset001_UCSFbrainmets/nnUNetTrainerMetsConfigurable_run5_1__nnUNetResEncUNetLPlans_custom_0624__3d_fullres/fold_0/training_log_2026_6_29_02_57_13.txt)

---

## 4. Scientific Analysis: The Impact of Extended Epochs (Run 5.1 vs Run 4.1)

Run 5.1 was designed to test whether expanding the training length from 300 epochs to 800 epochs (using the optimal hyperparameters of Run 4.1) would resolve the low Dice score issue on Tumor Core (TC) and Enhancing Tumor (ET). 

The results reveal a critical behavior:

1.  **Overall Loss Optimization Success:**
    Increasing epochs to 800 successfully drove the training loss down from $-0.4017$ (Run 4.1) to **$-0.4527$** (Run 5.1). The validation loss also improved, dropping from $-0.3188$ to **$-0.3243$**.
2.  **Whole Tumor (WT) Segmentation Improvement:**
    The Dice score of the Whole Tumor region (Region 1) improved from $61.57\%$ to **$64.57\%$**. This shows the model successfully consolidated its understanding of the macro structure of the metastases.
3.  **Severe Performance Degradation on TC and ET:**
    Despite the lower overall losses, the validation Dice score for **Tumor Core (TC) dropped from $24.52\%$ to $19.46\%$**, and **Enhancing Tumor (ET) dropped from $31.42\%$ to $21.43\%$**. The overall Mean Validation Dice subsequently declined from $39.17\%$ to $35.15\%$.

### Why did this happen? (Class Imbalance Overfitting Bias)
This is a standard symptom of volume class imbalance optimization over long schedules. 

The Whole Tumor region has a vastly larger volume footprint compared to the Enhancing Tumor and Tumor Core. In the beginning (up to 300 epochs), the optimizer (AdamW) updates parameters based on coarse gradients, enabling it to detect and segment both large and small parts relatively well (resulting in $31.42\%$ ET Dice). 

However, as training progresses to 800 epochs, the network concentrates on minimizing the dominant contributors to the loss—the boundaries of the large Whole Tumor region. Because the loss contribution of the tiny Enhancing Tumor region is extremely small (even with a region weight of `2.0`), the optimizer essentially sacrifices enhancing tumor detail to refine the whole tumor boundaries, leading to false negatives on small enhancing structures.

---

## 5. Strategic Directives for Upcoming Experiments

The findings from Run 5.1 validate our strategy that **simply training longer (increasing epochs) is counterproductive for small target segmentation** under class-imbalanced conditions. We must adjust the gradients during training.

### Immediate Focus (Configs already created in `exp5_0628`):
1.  **Adjust Region Weights (Run 5.3 & Run 5.4):**
    Evaluate if heavily penalizing small region errors by increasing weights to **`[1.0, 2.0, 5.0]`** preserves Enhancing Tumor (ET) and Tumor Core (TC) performance under a 300-epoch schedule.
2.  **Combine Weights & Slower Decay (Run 5.5):**
    Evaluate if combining the adjusted weights `[1.0, 2.0, 5.0]` with a slightly longer schedule of **600 epochs** allows the model to converge safely without sacrificing smaller regions.

### Secondary Focus (Loss Function Modifications):
If Run 5.3–5.5 do not yield ET Dice scores above $35\%$, we should transition to:
*   **Focal Loss integration:** Modify the base loss function inside `WeightedRegionLossWrapper` to compute a focal loss version of BCE for the Enhancing Tumor region, suppressing background voxel gradient dominance.
*   **Post-processing Connected Components:** Implement a volume filter to remove predicted segments below $10\text{ mm}^3$ (isolated voxels) to heavily reduce the false-positive rate.
