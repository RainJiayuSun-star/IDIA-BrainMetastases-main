# IDiA Lab - Preprocessing & Training Pipeline Report (June 22, 2026)

## 1. Summary
This report documents the recent end-to-end execution of our unified MRI preprocessing pipeline and the subsequent configuration of the nnU-Net v2 training environment. The objective was to standardize the **Brain-Mets-Lung** and **UCSF** datasets through skull-stripping, bias correction, and spatial alignment, and then launch a distributed multi-GPU training job to train a highly optimized brain metastases segmentation model using the newly released nnU-Net v2 framework. 

---

## 2. Preprocessing Pipeline

### 2.1 Datasets Processed
We successfully formatted and preprocessed two major cohorts for training:
* **UCSF Brain Mets Dataset**
* **Brain-Mets-Lung Dataset**

### 2.2 Methodology: rain_preprocess vs. BrambleScript_V2
During the Bias Correction phase, we evaluated two separate processing implementations. Both scripts rely on the underlying SimpleITK `N4BiasFieldCorrectionImageFilter` to mathematically eliminate low-frequency RF coil inhomogeneities (using a shrink factor of 2, 50 iterations, and a convergence threshold of 1e-6). 

**Differences & Use Cases:**
* **`logging_jbBiasCorrection_BrambleScript_V2.py`:** Jared's standalone script is strictly dependent on the pre-existence of highly accurate brain masks (expecting `_SS_bet.nii.gz` files from a prior skull-stripping step) to restrict the N4 calculation. 
* **`rain_preprocess.py` (Unified Pipeline):** We utilized this script because **our dataset did not have existing brain masks available.** To solve this, `rain_preprocess.py` implements an automatic fallback mechanism: when no mask is provided, it applies **Otsu Thresholding** (`sitk.OtsuThreshold`) directly to the image on the fly. 

**Impact on Quality:**
Otsu thresholding automatically calculates the optimal pixel intensity to separate the bright foreground tissue (the head/brain) from the dark background (the empty air surrounding the head). By forcing the N4 algorithm to only calculate the bias field inside this Otsu-generated foreground mask, the algorithm doesn't waste computational fitting power trying to model the noisy, empty air. This drastically improves the mathematical accuracy and quality of the bias field estimation inside the actual brain parenchyma.

---

## 3. Model Training Setup (nnU-Net v2)

Following preprocessing, we migrated the data into the strict `nnUnet_raw` folder structure (separating `imagesTr` and `labelsTr`) and initiated the nnU-Net v2 pipeline on the lab's virtual machine.

### 3.1 Fingerprinting and Experiment Planning
nnU-Net is a "self-configuring" method. We executed `nnUNetv2_plan_and_preprocess`, which scanned our preprocessed NIfTI files to extract dataset fingerprints (image dimensions, voxel spacings, and class ratios). 

**VRAM Optimization:** By default, nnU-Net plans for standard 8GB GPUs, which resulted in a severely restricted batch size that crashed PyTorch's distributed engine. Because our lab server utilizes four massive 40GB NVIDIA L40 GPUs, we injected the environment variable `-e nnUNet_vram_target_GB=35` into the Docker container. This forced the mathematical heuristic to drastically scale up the 3D Patch Size and Batch Size, maximizing our hardware utilization and solving the crash.

### 3.2 Multi-GPU Training Execution (DDP)
We initiated the training using the `3d_fullres` configuration. The training job was launched inside a customized Docker container with the following engineering optimizations:

1. **Distributed Data Parallel (DDP):** We utilized the `-num_gpus 2` (and eventually 4) flag to spread the massive 3D batches across the L40 GPUs.
2. **Shared Memory Bottleneck Resolution:** Multithreaded data augmentation in PyTorch often crashes Docker due to the default 64MB `/dev/shm` limit. We resolved this `[Errno 28] No space left on device` error by allocating 32GB of system RAM to shared memory (`--shm-size=32g`), ensuring the GPUs are continuously fed with data.
3. **PyTorch Compilation (JIT):** nnU-Net v2 enables PyTorch 2.0's `torch.compile` by default to mathematically fuse kernels and speed up training by up to 20%. To support this, we updated our `Dockerfile` to install the `gcc` and `build-essential` C-compilers, allowing the neural network graph to successfully compile on the fly.

### 3.3 Training Status (Mid-run Evaluation at Epoch 365)
The training job has run up to **epoch 365** (out of 1000). The mid-run evaluation reveals a clear performance plateau, indicating that the default single-stage configuration is approaching its architectural limits on this dataset.

#### Key Observations from the Curves
1. **Training Loss (loss_tr):** Shows a healthy, steady downward trajectory, dropping from roughly $-0.05$ to $-0.35$. The network is successfully learning and fitting the training distribution.
2. **Validation Loss (loss_val):** After a sharp drop in the first 25 epochs, it has plateaued in a highly volatile, oscillating band between $-0.17$ and $-0.25$ (averaging $\sim -0.21$). The validation loss is not increasing (meaning the model is not aggressively overfitting), but it has ceased generalized improvement.
3. **Pseudo-Dice Score:** The validation pseudo-dice moving average peaked around epoch 190 at **0.51** and has since slightly settled to **0.48–0.49** at epoch 365. The raw pseudo-dice scores oscillate wildly (between $0.30$ and $0.65$).

#### Diagnostic Analysis & Technical Explanations
* **Learning Rate Step-Size:** nnU-Net utilizes a poly learning rate decay schedule where $LR = LR_{initial} \times (1 - \frac{epoch}{max\_epoch})^{0.9}$. At epoch 365, the learning rate is still $\sim 66\%$ of its initial value. This relatively high learning rate causes the optimizer to take large steps, which yields massive fluctuations in validation loss and Dice scores when dealing with small, scattered targets.
* **Spatial Context Blindness:** By default, `3d_fullres` extracts small local patches (e.g., $128 \times 128 \times 128$) to fit standard hardware footprints. Because the model lacks a global view of the brain anatomy, it confuses normal contrast-enhancing structures (like blood vessels or the choroid plexus) with metastases, resulting in false positives.
* **Dice Loss Math on Small Targets:** Small metastases (under 5–10 mm) occupy very few voxels. A minor spatial misalignment of 1 or 2 pixels drops the individual lesion's Dice score to 0.40 or below. Consequently, global Dice optimization tends to prioritize large lesions and ignore smaller ones.

#### Next Steps
To mitigate these spatial and volumetric limitations and improve the baseline Dice score, we are initiating a parallel training run on the unallocated **GPU 2** with the following strategies:

1. **Residual Encoder Training (ResEnc):** Having executed the experiment planner using the `nnUNetPlannerResEncL` preset, we will train the `nnUNetResEncUNetLPlans` configuration. This replaces standard convolutional blocks with residual blocks to capture deeper, more robust semantic features. Note: The standard 3D Cascade Configuration is automatically omitted by nnU-Net's planner because our cropped brain volumes are small enough to fit within a single full-resolution patch.
2. **Manual Hyperparameter Optimization Experiments:** We will perform systematic ablation experiments by copying and manually editing the `nnUNetResEncUNetLPlans.json` configuration file to adjust:
   * **Batch Size Scaling:** Increasing the batch size from 2 to 4 or 8 to average gradients across more patches, stabilizing optimization in the presence of extreme class imbalance.
   * **Patch Size Expansion:** Scaling the patch dimensions (e.g., to $[160, 192, 160]$) to maximize the neural network's receptive field and provide the global anatomical context needed to reduce false positives.
   * **Downsampling Depth Preservation:** Restricting the number of pooling steps or modifying pooling strides at the bottleneck to prevent tiny (3–5 mm) metastases from being downsampled out of existence before the decoding layers.

### 3.4 Baseline Validation Results Analysis

We analyzed the validation performance of the baseline `3d_fullres` model (trained using the default plans file [plans.json](file:///Users/rains/rainworkspace/2-IDIALab/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset/nnUNet_results/Dataset001_UCSFbrainmets/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json) on Fold 0) by evaluating its validation metrics stored in [summary.json](file:///Users/rains/rainworkspace/2-IDIALab/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset/nnUNet_results/Dataset001_UCSFbrainmets/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation/summary.json).

* **Global Performance:** The overall mean foreground Dice score is low at **25.3%** (`Dice: 0.2533`).
* **Class-Specific Breakdown:**
  * **Class 1 (Edema / Surrounding Tissue):** Mean Dice score of **40.1%** (`Dice: 0.4013`).
  * **Class 2 (Non-enhancing core / Necrosis):** Mean Dice score of **20.3%** (`Dice: 0.2032`).
  * **Class 3 (Enhancing active tumor):** Mean Dice score of **15.5%** (`Dice: 0.1553`).
* **Major Diagnostic Findings:**
  1. **High False Positive Rate across Classes:** In several cases (e.g., `UCSFbrainmets_010`, `UCSFbrainmets_017`), the model predicted significant volumes of Class 1 or Class 2 when the reference masks had zero voxels for these classes. This indicates massive cross-class spatial confusion.
  2. **Severe False Negatives on Small Lesions (Class 3):** For patients with tiny enhancing metastases (Class 3, e.g., `UCSFbrainmets_011` with `n_ref = 75`), the model frequently outputs zero predictions (`n_pred = 0`). The default configuration completely misses these small targets, dominated by the gradients of larger classes.

### 3.5 Constructive Hyperparameter Tuning Plan

Based on the baseline diagnostic findings and the structural properties in [plans.json](file:///Users/rains/rainworkspace/2-IDIALab/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset/nnUNet_results/Dataset001_UCSFbrainmets/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json), we propose the following systematic hyperparameter tuning plan:

#### 1. Batch Size Scaling
* **Baseline:** `batch_size: 2`
* **Proposed Tuning:** Increase to `batch_size: 4` or `batch_size: 8`.
* **Rationale:** A batch size of 2 produces highly noisy gradients, especially in highly class-imbalanced volumes. Scaling to 4 or 8 will stabilize optimization and prevent the model from plateauing or ignoring small classes.

#### 2. Patch Size Extension
* **Baseline:** `patch_size: [80, 192, 160]` (vs. Median Image Size `[91, 202, 160]`)
* **Proposed Tuning:** Expand patch size to `[96, 224, 160]`.
* **Rationale:** The baseline patch size is close to the median volume size but still clips the boundaries. Fully covering the brain structure prevents spatial context truncation and reduces false positives at patch borders.

#### 3. Loss Function & Class Weighting
* **Baseline:** Equal weighting of Cross-Entropy and Soft Dice loss.
* **Proposed Tuning:** Implement **Region-Based Training** by grouping labels into overlapping regions:
  * *Whole Tumor (WT):* Labels 1 + 2 + 3
  * *Tumor Core (TC):* Labels 2 + 3
  * *Enhancing Tumor (ET):* Label 3
* **Rationale:** This removes the strict mutual exclusion constraints that cause cross-class confusion, allowing the model to learn nested anatomical hierarchies.

#### 4. Spatial Downsampling Optimization
* **Baseline Strides:** `[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]`
* **Proposed Tuning:** Modify the last stage stride to prevent aggressive downsampling.
* **Rationale:** Preserving spatial resolution at the bottleneck prevents the geometry of tiny (3–5 mm) metastases from being reduced to a single voxel, ensuring the decoder has fine details to reconstruct.

#### 5. Regularization & Optimizer Tuning
* **Baseline:** SGD optimizer with learning rate `0.01` and weight decay `3e-5`.
* **Proposed Tuning:** Transition to **AdamW** with an initial learning rate of **`3e-4`** (utilizing the pre-built `nnUNetTrainerAdam3en4` class) and increase weight decay to **`1e-4`** or **`1e-3`**.
* **Rationale:** The baseline SGD with light weight decay allows the network's weights to grow unchecked, leading to memorization of training-specific background noise. Increasing weight decay penalizes large weights, while AdamW with a smaller learning rate stabilizes convergence on sparse, multi-focal targets.

#### 6. Spatial Dropout Activation
* **Baseline:** Disabled (`"dropout_op": null`).
* **Proposed Tuning:** Enable 3D Spatial Dropout with a rate of `0.2` or `0.3` in the encoder path.
* **Rationale:** Actively dropping entire feature maps during training breaks up channel co-adaptation, forcing the model to learn more robust, generalized features and mitigating the severe validation plateau.

#### 7. Early Stopping / Training Length Control
* **Baseline:** Hardcoded 1000 epochs.
* **Proposed Tuning:** Reduce training length to **250 or 500 epochs** (using the pre-built `nnUNetTrainer_250epochs` or `nnUNetTrainer_500epochs` wrappers).
* **Rationale:** The moving average pseudo Dice peaks and validation loss completely plateaus by epoch 200–300. The remaining 700 epochs contribute solely to overfitting the training set (shown by the training loss continuously decreasing to $-0.43$). Shortening the schedule acts as a form of early stopping, reducing over-parameterization and compute time.

---


## 4. Other Promising Architectures

Beyond the standard and residual-encoder configurations of nnU-Net, several specialized neural network architectures have shown promising results for brain metastases segmentation, particularly in international benchmarks such as the BraTS-METS challenge:

1. **Swin UNETR (Shifted Window Transformer U-Net):**
   * **Mechanism:** Integrates a Swin Transformer as the encoder to extract multi-scale, long-range global context, while utilizing a convolutional (CNN) decoder to reconstruct high-resolution boundary maps.
   * **Clinical Rationale:** The global receptive field provides the network with context regarding where a patch resides inside the entire brain volume, reducing false-positive segmentations in anatomically similar enhancing structures (such as healthy blood vessels or the choroid plexus).

2. **3D UX-Net (Large-Kernel CNN):**
   * **Mechanism:** Replaces typical $3 \times 3 \times 3$ convolutions with large convolutional kernels (e.g., $7 \times 7 \times 7$), drawing architectural inspiration from ConvNeXt.
   * **Clinical Rationale:** It replicates the expansive spatial receptive fields of Transformers while maintaining the superior inductive bias, speed, and efficiency of pure convolutional neural networks.

3. **Deformable Convolutional Networks (e.g., 3D-MedDCNet):**
   * **Mechanism:** Utilizes deformable convolutional layers that learn 3D offsets to shape-adaptively distort the kernel's sampling grid.
   * **Clinical Rationale:** Since brain metastases exhibit highly irregular, heterogeneous shapes and arbitrary boundaries, deformable convolutions align more precisely with lesion contours than rigid grid convolutions.

4. **Attention U-Net:**
   * **Mechanism:** Introduces attention gates (AGs) in the skip connections.
   * **Clinical Rationale:** AGs automatically suppress feature activations in background regions and healthy tissues, directing model capacity to focus specifically on enhancing lesion boundaries.

5. **Loss Function Optimizations:**
   * **Mechanism:** Augmenting standard Soft Dice loss with **Focal Tversky Loss** or **Boundary Loss**.
   * **Clinical Rationale:** These losses place higher gradient penalties on false negatives (missed tiny lesions) and boundary errors, mitigating the bias of global Dice toward large metastases.

## 5. Current State of the Art Models

Based on recent competitive leaderboards (such as the BraTS-METS challenge proceedings), the current top-performing paradigms for brain metastases segmentation are:

1. **MedNeXt (State-of-the-Art 3D CNN):**
   * **Mechanism:** Redesigns the standard 3D U-Net using blocks inspired by **ConvNeXt**, utilizing depthwise convolutions, residual connections, and large kernel sizes. Modern variants like **Fast-MedNeXt** optimize this with partial convolutions (PConv) to reduce GPU latency.
   * **Clinical Rationale:** It preserves the strong local spatial inductive biases of CNNs (essential for finding voxel-level metastases) while expanding the spatial receptive field.

2. **SegMamba (Mamba-based State Space Model):**
   * **Mechanism:** Integrates the **Mamba** state space model (SSM) architecture instead of self-attention.
   * **Clinical Rationale:** By modeling 3D volume inputs with **linear complexity** (relative to image size), SegMamba handles massive 3D patches or entire volumes efficiently, capturing long-range spatial sequence context to differentiate multifocal lesions from healthy blood vessels.

3. **3D-TransUNet (Transformer-CNN Hybrid):**
   * **Mechanism:** Pairs a Vision Transformer (ViT) encoder with a 3D U-Net CNN decoder.
   * **Clinical Rationale:** In the **BraTS-METS 2023 Challenge**, an encoder-only 3D-TransUNet configuration achieved **2nd place** overall on the test cohort, reporting an average **lesion-wise Dice score of 59.8%**. The ViT encoder handles large-scale contextual reasoning, while the CNN decoder structures details.

4. **Cross-Architecture Ensembling (State-of-the-Art Challenge Paradigm):**
   * **Mechanism:** Combining the softmax probability maps of multiple complementary models (e.g., nnU-Net + MedNeXt + Swin UNETR).
   * **Clinical Rationale:** Top-performing teams in the BraTS challenges rely on ensembles to combine the local accuracy of CNNs with the global consistency of Transformers, which consistently yields a 3–5% increase in final lesion-wise Dice accuracy.
