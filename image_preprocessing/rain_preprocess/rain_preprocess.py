# -*- coding: utf-8 -*-
"""
IDiA Lab Unified Preprocessing Pipeline
============================================================
A highly robust, configurable, and reusable pipeline that wraps co-registration,
skull stripping (HD-BET), N4 bias field correction, template space registration,
and percentile intensity normalization into a single, unified execution chain.

Requirements:
    - SimpleITK, pandas, numpy (Standard clinical imaging libraries)
    - PyYAML (Optional, with custom native YAML parser fallback built-in)
    - HD-BET (Required only if running Step 2: Skull Stripping)
"""

import os
import sys
import time
import argparse
import gc
import json
import shutil
import traceback
import platform
import importlib.metadata
import numpy as np
import pandas as pd
import SimpleITK as sitk

# Try importing YAML, fallback to custom parser if missing
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Try importing torch/HD-BET, warn if missing (only needed for Step 2)
try:
    import torch
    from HD_BET.hd_bet_prediction import hdbet_predict
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from HD_BET.paths import folder_with_parameter_files
    HAS_HDBET = True
except ImportError:
    HAS_HDBET = False


# ==========================================
# 0. Custom Native YAML Parser Fallback
# ==========================================
def parse_yaml_fallback(filepath):
    """
    A lightweight, zero-dependency parser for YAML files.
    Allows loading configurations without PyYAML installed.
    Respects indentation so nested blocks (n4_params / in_params) do not
    swallow subsequent top-level keys like patient_range / num_workers.
    """
    config = {}
    current_key = None
    with open(filepath, 'r') as f:
        for raw_line in f:
            # Preserve leading whitespace to detect nesting; strip trailing comments/newlines
            if '#' in raw_line:
                # Only strip comments outside quoted strings (simple heuristic)
                hash_idx = raw_line.find('#')
                # Treat full-line comments and trailing inline comments
                if raw_line.lstrip().startswith('#'):
                    continue
                # Inline comment: keep content before ' #'
                if ' #' in raw_line:
                    raw_line = raw_line.split(' #', 1)[0]
            if not raw_line.strip():
                continue

            indent = len(raw_line) - len(raw_line.lstrip(' '))
            line = raw_line.strip()

            if ':' in line:
                parts = line.split(':', 1)
                k = parts[0].strip()
                v = parts[1].strip()

                # Check if it starts a list or dict block
                if not v:
                    current_key = k
                    config[k] = [] if k == 'steps' else {}
                else:
                    # Parse basic types
                    if v.lower() == 'true':
                        v = True
                    elif v.lower() == 'false':
                        v = False
                    elif v.lower() == 'none':
                        v = None
                    elif v.startswith('[') and v.endswith(']'):
                        # Parse lists like [3, 6], [21, 91], or ["a", "b"]
                        items = [x.strip() for x in v[1:-1].split(',') if x.strip()]
                        parsed = []
                        for item in items:
                            if (item.startswith('"') and item.endswith('"')) or (item.startswith("'") and item.endswith("'")):
                                parsed.append(item[1:-1])
                            else:
                                try:
                                    parsed.append(int(item) if '.' not in item else float(item))
                                except ValueError:
                                    parsed.append(item)
                        v = parsed
                    else:
                        try:
                            if '.' in v:
                                v = float(v)
                            else:
                                v = int(v)
                        except ValueError:
                            # Keep as clean string
                            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                                v = v[1:-1]

                    # Nested keys are indented; top-level keys reset the block
                    if indent > 0 and current_key and isinstance(config.get(current_key), dict):
                        config[current_key][k] = v
                    else:
                        config[k] = v
                        current_key = None
            elif line.startswith('-') and current_key == 'steps':
                val = line[1:].strip()
                if val:
                    config['steps'].append(int(val))
    return config


# ==========================================
# 1. Pipeline Action Functions
# ==========================================

def step_1_coregistration(img_path, ref_path, output_path):
    """
    Step 1: Mutual-Information-based Rigid Co-registration.
    Aligns a moving image (FLAIR/Mask) to a fixed baseline (T1CE/T1post).
    """
    print(f"      [Step 1] Aligning {os.path.basename(img_path)} -> {os.path.basename(ref_path)}")
    fixed = sitk.ReadImage(ref_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(img_path, sitk.sitkFloat32)
    
    # Standard mutual info rigid registration configuration
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.15)
    R.SetInterpolator(sitk.sitkLinear)
    
    # Rigid transformation (6 degrees of freedom in 3D) center-initialized
    tx = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    R.SetInitialTransform(tx, inPlace=True)
    R.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, estimateLearningRate=R.EachIteration)
    R.SetOptimizerScalesFromPhysicalShift()
    
    # Run optimization
    out_tx = R.Execute(fixed, moving)
    
    # Resample moving image to fixed reference grid
    aligned = sitk.Resample(moving, fixed, out_tx, sitk.sitkLinear, 0.0, moving.GetPixelID())
    sitk.WriteImage(aligned, output_path)
    print(f"      [Step 1 Saved] Co-registered scan: {os.path.basename(output_path)}")


def step_2_skull_stripping(img_path, output_img, output_mask, device="cpu"):
    """
    Step 2: HD-BET Skull Stripping and Brain Extraction.
    """
    if not HAS_HDBET:
        raise ImportError("HD-BET package or dependencies not installed. Ensure torch and hd-bet are in your environment.")
    
    print(f"      [Step 2] Skull Stripping {os.path.basename(img_path)}")
    dev = torch.device(device)
    
    # Setup custom predictor matching lab implementation
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=(device != "cpu"),
        device=dev,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(
        folder_with_parameter_files, 
        use_folds='all',
        checkpoint_name='checkpoint_final.pth'
    )
    
    # Execute prediction
    hdbet_predict(img_path, output_img, predictor, keep_brain_mask=True)
    
    # Re-save the mask to the correct output path if needed
    hdbet_mask_default = output_img.replace(".nii.gz", "_bet.nii.gz")
    if os.path.exists(hdbet_mask_default):
        if hdbet_mask_default != output_mask:
            os.rename(hdbet_mask_default, output_mask)
        print(f"      [Step 2 Saved] Extracted Brain: {os.path.basename(output_img)}")
        print(f"      [Step 2 Saved] Brain Mask: {os.path.basename(output_mask)}")
    else:
        print(f"      [Warning] Mask not automatically found at {hdbet_mask_default}")


def step_3_n4_bias_correction(img_path, mask_path, output_path, shrink_factor=2, num_iters=50, convergence_thresh=1e-6):
    """
    Step 3: N4 Bias Field Correction using SimpleITK.
    """
    print(f"      [Step 3] Running N4 Bias Correction on {os.path.basename(img_path)}")
    img = sitk.ReadImage(img_path, sitk.sitkFloat32)
    
    # Read mask if available; otherwise generate a simple bounding mask
    if mask_path and os.path.exists(mask_path):
        mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)
    else:
        print("      [Warning] No skull mask found. Generating automatic foreground mask...")
        mask = sitk.OtsuThreshold(img, 0, 1)
        
    # Shrunk image for faster bias estimation
    if shrink_factor > 1:
        img_shrunk = sitk.Shrink(img, [shrink_factor] * img.GetDimension())
        mask_shrunk = sitk.Shrink(mask, [shrink_factor] * img.GetDimension())
    else:
        img_shrunk, mask_shrunk = img, mask
        
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([num_iters] * 4)
    corrector.SetConvergenceThreshold(convergence_thresh)
    corrector.Execute(img_shrunk, mask_shrunk)
    
    log_bias_field = corrector.GetLogBiasFieldAsImage(img)
    corrected_img = img / sitk.Exp(log_bias_field)
    
    sitk.WriteImage(corrected_img, output_path)
    print(f"      [Step 3 Saved] Bias Corrected: {os.path.basename(output_path)}")


def step_5_template_warp(ref_path, template_path, output_ref_path, other_images=None):
    """
    Step 5: Diffeomorphic/Affine Registration to standard Template Space (MNI152 or SRI-24).
    Computes registration transform on the reference image (T1CE/T1post) and resamples all images
    to the template space. Automatically uses Nearest Neighbor interpolation for masks/labels
    to preserve integer values, and Linear interpolation for structural scans.
    """
    print(f"      [Step 5] Registering Reference {os.path.basename(ref_path)} -> {os.path.basename(template_path)}")
    fixed = sitk.ReadImage(template_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(ref_path, sitk.sitkFloat32)
    
    # Rigid + Affine Multi-resolution registration to Template
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.10)
    R.SetInterpolator(sitk.sitkLinear)
    
    # Affine transformation center-initialized to align templates in physical space
    tx = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.AffineTransform(moving.GetDimension()),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    R.SetInitialTransform(tx, inPlace=True)
    R.SetOptimizerAsGradientDescent(learningRate=0.5, numberOfIterations=80, estimateLearningRate=R.EachIteration)
    R.SetOptimizerScalesFromPhysicalShift()
    
    # Compute transform
    out_tx = R.Execute(fixed, moving)
    
    # Warp reference scan
    warped_ref = sitk.Resample(moving, fixed, out_tx, sitk.sitkLinear, 0.0, moving.GetPixelID())
    sitk.WriteImage(warped_ref, output_ref_path)
    print(f"      [Step 5 Saved] Warped Reference T1-CE: {os.path.basename(output_ref_path)}")
    
    # Warp other modalities and segmentations using the computed transform
    if other_images:
        for item in other_images:
            img_p = item['path']
            out_p = item['out']
            is_mask = item.get('is_mask', False)
            
            if not os.path.exists(img_p):
                continue
                
            print(f"      [Step 5 Applying Transform] Resampling {os.path.basename(img_p)}...")
            moving_other = sitk.ReadImage(img_p, sitk.sitkFloat32)
            
            # Select proper interpolation
            interp = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
            
            # Resample and write
            warped_other = sitk.Resample(moving_other, fixed, out_tx, interp, 0.0, moving_other.GetPixelID())
            sitk.WriteImage(warped_other, out_p)
            print(f"      [Step 5 Saved] Warped Scan: {os.path.basename(out_p)}")


def step_6_intensity_normalization(img_path, mask_path, output_path, lower_p=0.01, upper_p=0.99, target_max=4000.0):
    """
    Step 6: Percentile-based Intensity Normalization.
    """
    print(f"      [Step 6] Normalizing intensities on {os.path.basename(img_path)}")
    img = sitk.ReadImage(img_path, sitk.sitkFloat32)
    
    if mask_path and os.path.exists(mask_path):
        mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)
    else:
        mask = sitk.OtsuThreshold(img, 0, 1)
        
    img_array = sitk.GetArrayFromImage(img)
    mask_array = sitk.GetArrayFromImage(mask)
    brain_voxels = img_array[mask_array > 0]
    
    p_low = pd.Series(brain_voxels).quantile(lower_p)
    p_high = pd.Series(brain_voxels).quantile(upper_p)
    
    normalized_img = (img - p_low) * (target_max / (p_high - p_low))
    normalized_img = sitk.Clamp(normalized_img, sitk.sitkFloat32, 0.0, target_max)
    
    sitk.WriteImage(normalized_img, output_path)
    print(f"      [Step 6 Saved] Normalized: {os.path.basename(output_path)}")


# ==========================================
# 2. Main Orchestrator Loop
# ==========================================

def preprocess_single_patient(pid, idx, total_patients, input_dir, output_dir, steps, device, intermediates_root, n4_params, in_params, template_path):
    print(f"\n[{idx+1}/{total_patients}] Processing Patient: {pid} ...")
    patient_start = time.time()
    
    try:
        # Core modality mapping for Yale / UCSF matching
        p_in_dir = os.path.join(input_dir, pid)
        all_p_files = os.listdir(p_in_dir)
        
        # 1. Discover Brain Mask (if already present)
        mask_file = next((f for f in all_p_files if ("bet" in f.lower() or "brain_mask" in f.lower()) and f.endswith(".nii.gz")), None)
        if not mask_file:
            mask_file = next((f for f in all_p_files if "mask" in f.lower() and "seg" not in f.lower() and f.endswith(".nii.gz")), None)
        
        # 2. Discover Tumor Segmentations (containing "seg" or "tumor")
        seg_files = [
            f for f in all_p_files 
            if ("seg" in f.lower() or "tumor" in f.lower()) 
            and f.endswith(".nii.gz") 
            and f != mask_file
        ]
        
        # 3. Discover Reference Scan (T1CE or T1post)
        t1ce_file = next((f for f in all_p_files if ("t1ce" in f.lower() or "t1post" in f.lower()) and f.endswith(".nii.gz")), None)
        
        # 4. Discover all other structural modalities (FLAIR, T1pre, T2Synth, T2, etc.)
        other_structural_files = [
            f for f in all_p_files 
            if f.endswith(".nii.gz")
            and f != t1ce_file
            and f != mask_file
            and f not in seg_files
            and "subtraction" not in f.lower()
        ]
        
        # Fallback if no specific naming is found
        if not t1ce_file and len(all_p_files) > 0:
            available_candidates = [
                f for f in all_p_files 
                if f.endswith(".nii.gz") 
                and f not in seg_files 
                and f != mask_file 
                and "subtraction" not in f.lower()
            ]
            if available_candidates:
                t1ce_file = available_candidates[0]
                other_structural_files = [f for f in available_candidates if f != t1ce_file]
        
        if not t1ce_file:
            print(f"   [{pid}] [Warning] No readable structural NIfTI scans found. Skipping.")
            return None
            
        print(f"   [{pid}] Detected Sequences:")
        print(f"   [{pid}]    - Reference/T1-CE:   {t1ce_file}")
        print(f"   [{pid}]    - Other Structurals: {other_structural_files if other_structural_files else 'None'}")
        print(f"   [{pid}]    - Brain Mask:        {mask_file if mask_file else 'N/A'}")
        print(f"   [{pid}]    - Segmentations:     {seg_files if seg_files else 'None'}")
        
        # Setup intermediate tracks
        current_t1_path = os.path.join(p_in_dir, t1ce_file)
        current_other_structural_paths = {f: os.path.join(p_in_dir, f) for f in other_structural_files}
        current_mask_path = os.path.join(p_in_dir, mask_file) if mask_file else None
        current_seg_paths = {sf: os.path.join(p_in_dir, sf) for sf in seg_files}
        
        # ==========================================
        # RUN STEPS SEQUENTIALLY
        # ==========================================
        
        # --- STEP 1: CO-REGISTRATION ---
        if 1 in steps:
            print(f"   [{pid}] >> Running Step 1: Modality Co-registration...")
            step_dir = os.path.join(intermediates_root, "step_1", pid)
            os.makedirs(step_dir, exist_ok=True)
            
            for f, f_path in current_other_structural_paths.items():
                out_aligned = os.path.join(step_dir, f.replace(".nii.gz", "_aligned.nii.gz"))
                step_1_coregistration(f_path, current_t1_path, out_aligned)
                current_other_structural_paths[f] = out_aligned
        
        # --- STEP 2: SKULL STRIPPING ---
        if 2 in steps:
            print(f"   [{pid}] >> Running Step 2: HD-BET Skull Stripping...")
            step_dir = os.path.join(intermediates_root, "step_2", pid)
            os.makedirs(step_dir, exist_ok=True)
            
            out_ss_t1 = os.path.join(step_dir, t1ce_file.replace(".nii.gz", "_SS.nii.gz"))
            out_mask = os.path.join(step_dir, t1ce_file.replace(".nii.gz", "_SS_bet.nii.gz"))
            
            step_2_skull_stripping(current_t1_path, out_ss_t1, out_mask, device=device)
            
            current_t1_path = out_ss_t1
            current_mask_path = out_mask
            
            for f, f_path in current_other_structural_paths.items():
                img = sitk.ReadImage(f_path, sitk.sitkFloat32)
                brain_mask = sitk.ReadImage(current_mask_path, sitk.sitkUInt8)
                stripped = sitk.Mask(img, brain_mask)
                
                out_ss_other = os.path.join(step_dir, f.replace(".nii.gz", "_SS.nii.gz"))
                sitk.WriteImage(stripped, out_ss_other)
                current_other_structural_paths[f] = out_ss_other
        
        # --- STEP 3: N4 BIAS CORRECTION ---
        if 3 in steps:
            print(f"   [{pid}] >> Running Step 3: N4 Bias Field Correction...")
            step_dir = os.path.join(intermediates_root, "step_3", pid)
            os.makedirs(step_dir, exist_ok=True)
            
            # Bias correct reference T1CE
            out_bc_t1 = os.path.join(step_dir, os.path.basename(current_t1_path).replace(".nii.gz", "_BC.nii.gz"))
            step_3_n4_bias_correction(
                current_t1_path, current_mask_path, out_bc_t1,
                shrink_factor=n4_params['shrink_factor'],
                num_iters=n4_params['num_iters'],
                convergence_thresh=n4_params['convergence_thresh']
            )
            current_t1_path = out_bc_t1
            
            # Bias correct all other structural scans
            for f, f_path in current_other_structural_paths.items():
                out_bc_other = os.path.join(step_dir, os.path.basename(f_path).replace(".nii.gz", "_BC.nii.gz"))
                step_3_n4_bias_correction(
                    f_path, current_mask_path, out_bc_other,
                    shrink_factor=n4_params['shrink_factor'],
                    num_iters=n4_params['num_iters'],
                    convergence_thresh=n4_params['convergence_thresh']
                )
                current_other_structural_paths[f] = out_bc_other
        
        # --- STEP 5: TEMPLATE SPACE WARP ---
        if 5 in steps:
            print(f"   [{pid}] >> Running Step 5: Template Space Warping...")
            if template_path and os.path.exists(template_path):
                step_dir = os.path.join(intermediates_root, "step_5", pid)
                os.makedirs(step_dir, exist_ok=True)
                
                other_images = []
                
                # Add all other structural scans
                warp_structural_paths = {}
                for f, f_path in current_other_structural_paths.items():
                    out_warp_other = os.path.join(step_dir, os.path.basename(f_path).replace(".nii.gz", "_MNI.nii.gz"))
                    other_images.append({'path': f_path, 'out': out_warp_other, 'is_mask': False})
                    warp_structural_paths[f] = out_warp_other
                    
                # Add Brain Mask
                out_warp_mask = None
                if current_mask_path:
                    out_warp_mask = os.path.join(step_dir, os.path.basename(current_mask_path).replace(".nii.gz", "_MNI.nii.gz"))
                    other_images.append({'path': current_mask_path, 'out': out_warp_mask, 'is_mask': True})
                    
                # Add all tumor segmentations
                warp_seg_paths = {}
                for sf, sf_path in current_seg_paths.items():
                    out_warp_seg = os.path.join(step_dir, sf.replace(".nii.gz", "_MNI.nii.gz"))
                    other_images.append({'path': sf_path, 'out': out_warp_seg, 'is_mask': True})
                    warp_seg_paths[sf] = out_warp_seg
                
                # Perform Warp
                out_warp_t1 = os.path.join(step_dir, os.path.basename(current_t1_path).replace(".nii.gz", "_MNI.nii.gz"))
                step_5_template_warp(current_t1_path, template_path, out_warp_t1, other_images=other_images)
                
                # Update active paths
                current_t1_path = out_warp_t1
                for f in current_other_structural_paths:
                    current_other_structural_paths[f] = warp_structural_paths[f]
                if current_mask_path:
                    current_mask_path = out_warp_mask
                for sf in current_seg_paths:
                    current_seg_paths[sf] = warp_seg_paths[sf]
            else:
                print(f"   [{pid}] [Warning] [Skipping Step 5] Valid template_path reference was not provided in the YAML configuration.")
        
        # --- STEP 6: INTENSITY NORMALIZATION ---
        if 6 in steps:
            print(f"   [{pid}] >> Running Step 6: Percentile Intensity Normalization...")
            step_dir = os.path.join(intermediates_root, "step_6", pid)
            os.makedirs(step_dir, exist_ok=True)
            
            # Normalize Reference T1CE
            out_norm_t1 = os.path.join(step_dir, os.path.basename(current_t1_path).replace(".nii.gz", "_IN.nii.gz"))
            step_6_intensity_normalization(
                current_t1_path, current_mask_path, out_norm_t1,
                lower_p=in_params['lower_percentile'],
                upper_p=in_params['upper_percentile'],
                target_max=in_params['target_max']
            )
            current_t1_path = out_norm_t1
            
            # Normalize all other structural scans
            for f, f_path in current_other_structural_paths.items():
                out_norm_other = os.path.join(step_dir, os.path.basename(f_path).replace(".nii.gz", "_IN.nii.gz"))
                step_6_intensity_normalization(
                    f_path, current_mask_path, out_norm_other,
                    lower_p=in_params['lower_percentile'],
                    upper_p=in_params['upper_percentile'],
                    target_max=in_params['target_max']
                )
                current_other_structural_paths[f] = out_norm_other
        
        # ==========================================
        # MOVE FINAL PRODUCTS TO OUTPUT ROOT
        # ==========================================
        p_out_final_dir = os.path.join(output_dir, pid)
        os.makedirs(p_out_final_dir, exist_ok=True)
        
        # Save Reference T1CE
        t1_final_name = os.path.basename(current_t1_path)
        sitk.WriteImage(sitk.ReadImage(current_t1_path), os.path.join(p_out_final_dir, t1_final_name))
        print(f"   [{pid}] [Final Product Saved] T1-CE (Reference): {t1_final_name}")
        
        # Save all other structural scans
        for f, f_path in current_other_structural_paths.items():
            other_final_name = os.path.basename(f_path)
            sitk.WriteImage(sitk.ReadImage(f_path), os.path.join(p_out_final_dir, other_final_name))
            print(f"   [{pid}] [Final Product Saved] Structural Scan:    {other_final_name}")
            
        if current_mask_path:
            mask_final_name = os.path.basename(current_mask_path)
            sitk.WriteImage(sitk.ReadImage(current_mask_path), os.path.join(p_out_final_dir, mask_final_name))
            print(f"   [{pid}] [Final Product Saved] Mask:               {mask_final_name}")
            
        # Save all segmentations
        for sf, sf_path in current_seg_paths.items():
            seg_final_name = os.path.basename(sf_path)
            sitk.WriteImage(sitk.ReadImage(sf_path), os.path.join(p_out_final_dir, seg_final_name))
            print(f"   [{pid}] [Final Product Saved] Tumor Segmentation:  {seg_final_name}")
            
        # Safely copy unprocessed files
        processed_basenames = [os.path.basename(current_t1_path)] + \
                              [os.path.basename(f_path) for f_path in current_other_structural_paths.values()] + \
                              ([os.path.basename(current_mask_path)] if current_mask_path else []) + \
                              [os.path.basename(sf_path) for sf_path in current_seg_paths.values()]
                              
        for f in all_p_files:
            is_processed = False
            for pb in processed_basenames:
                pb_clean = pb.replace("_MNI", "").replace("_BC", "").replace("_aligned", "").replace("_IN", "").replace("_SS", "")
                if f.split('.')[0] in pb_clean or pb_clean.split('.')[0] in f:
                    is_processed = True
                    break
            
            if not is_processed:
                src_f = os.path.join(p_in_dir, f)
                dest_f = os.path.join(p_out_final_dir, f)
                if os.path.isfile(src_f) and not os.path.exists(dest_f):
                    shutil.copy2(src_f, dest_f)
                    print(f"   [{pid}] [Synchronized] Copied raw file as-is: {f}")
                    
        elapsed = time.time() - patient_start
        print(f"   [{pid}] [Done] Successfully preprocessed in {elapsed:.1f}s.")
        return None
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"   [{pid}] [ERROR] Failed to preprocess patient: {error_msg}")
        traceback.print_exc(file=sys.stdout)
        return {"patient": pid, "error": error_msg}
        
    finally:
        gc.collect()


def run_pipeline(config_path):
    print("==================================================")
    print("      IDiA UNIFIED PREPROCESSING PIPELINE         ")
    print("==================================================\n")
    
    # 1. Load config
    if HAS_YAML:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded YAML configuration from PyYAML: {config_path}")
    else:
        config = parse_yaml_fallback(config_path)
        print(f"Loaded YAML configuration via custom fallback: {config_path}")
        
    input_dir = config.get('input_dir')
    output_dir = config.get('output_dir')
    steps = config.get('steps', [1, 2, 3, 5, 6])
    device = config.get('device', 'cpu')
    num_workers = config.get('num_workers', 1)
    
    # Algorithm details
    n4_params = config.get('n4_params', {'shrink_factor': 2, 'num_iters': 50, 'convergence_thresh': 1e-6})
    in_params = config.get('in_params', {'lower_percentile': 0.01, 'upper_percentile': 0.99, 'target_max': 4000.0})
    template_path = config.get('template_path', '')
    
    print("\n--- Configurations ---")
    print(f"  Input Root:  {input_dir}")
    print(f"  Output Root: {output_dir}")
    print(f"  Selected Pipeline Steps: {steps}")
    print(f"  Compute Device: {device}")
    print(f"  Parallel Workers: {num_workers}")
    print(f"  Template Reference: {template_path if template_path else 'None (Required for Step 5)'}")
    print("----------------------\n")
    
    if not os.path.exists(input_dir):
        print(f"FATAL: Input directory '{input_dir}' does not exist.")
        sys.exit(1)
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Intermediate directories setup
    intermediates_root = os.path.join(output_dir, "intermediates")
    os.makedirs(intermediates_root, exist_ok=True)
    
    # Discover patient directories
    all_entries = os.listdir(input_dir)
    patient_ids = sorted([
        d for d in all_entries 
        if os.path.isdir(os.path.join(input_dir, d)) and not d.startswith('.')
    ])
    
    print(f"Discovered {len(patient_ids)} candidate patient folders.")
    
    # Optional patient subset selection
    patient_list = config.get('patient_list')
    patient_range = config.get('patient_range')
    patient_limit = config.get('patient_limit')
    
    if patient_list:
        if isinstance(patient_list, str):
            patient_list = [patient_list]
        patient_ids = [pid for pid in patient_ids if pid in patient_list]
        print(f"   [Subset Selection] Selected {len(patient_ids)} patient(s) specified in 'patient_list'.")
        
    elif patient_range:
        if isinstance(patient_range, list) and len(patient_range) == 2:
            start_idx = max(1, int(patient_range[0])) - 1
            end_idx = min(len(patient_ids), int(patient_range[1]))
            patient_ids = patient_ids[start_idx:end_idx]
            print(f"   [Subset Selection] Selected patient range {patient_range[0]} to {patient_range[1]} (Total: {len(patient_ids)}).")
            
    elif patient_limit:
        limit = int(patient_limit)
        patient_ids = patient_ids[:limit]
        print(f"   [Subset Selection] Limited execution to the first {limit} patients.")
        
    start_total_time = time.time()
    errors_list = []
    
    if num_workers > 1:
        import concurrent.futures
        print(f"Launching multi-process processing pool with {num_workers} workers...")
        # Set SimpleITK to single-threaded within workers to avoid thread oversubscription
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
        
        futures = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            for idx, pid in enumerate(patient_ids):
                futures.append(executor.submit(
                    preprocess_single_patient,
                    pid, idx, len(patient_ids),
                    input_dir, output_dir, steps, device, intermediates_root,
                    n4_params, in_params, template_path
                ))
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    if res:
                        errors_list.append(res)
                except Exception as e:
                    print(f"   [Worker Crash] Worker raised exception: {e}")
    else:
        # Sequential Execution
        for idx, pid in enumerate(patient_ids):
            res = preprocess_single_patient(
                pid, idx, len(patient_ids),
                input_dir, output_dir, steps, device, intermediates_root,
                n4_params, in_params, template_path
            )
            if res:
                errors_list.append(res)
                
    # Print ending summary
    total_time = time.time() - start_total_time
    print("\n==================================================")
    print("--- PIPELINE ENDING SUMMARY ---")
    print(f"Total patient files attempted: {len(patient_ids)}")
    print(f"Total errors: {len(errors_list)}")
    print(f"Total execution time: {total_time:.1f}s")
    if errors_list:
        print("\nErrors Logged:")
        for err in errors_list:
            print(f"  Patient {err['patient']}: {err['error']}")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDiA Lab Unified Preprocessing Pipeline")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "config_pos",
        nargs="?",
        type=str,
        default=None,
        help="Path to the YAML configuration file (positional argument fallback)"
    )
    
    args = parser.parse_args()
    
    # Prioritize positional argument for backwards compatibility, otherwise use --config
    config_file = args.config_pos if args.config_pos else args.config
    
    if not os.path.exists(config_file):
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)
        
    run_pipeline(config_file)
