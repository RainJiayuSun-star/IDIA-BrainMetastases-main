# -*- coding: utf-8 -*-
"""
IDiA Lab - Step 3 N4 Bias Field Correction Pipeline (Single-Worker Edition)
=============================================================================
File: rain_preprocess_step3_bc.py

Combines Jared's robust logging framework (logging_jbBiasCorrection_BrambleScript_V2.py)
with rain_preprocess.py's automatic Otsu thresholding fallback for datasets without 
pre-computed brain masks.

Features:
  1. PDF Logging & Excel Error Tracking (fpdf2 / pandas).
  2. Automatic Brain Mask Fallback: Uses existing brain masks if available (e.g. _SS_bet.nii.gz, 
     _mask.nii.gz), otherwise automatically computes an Otsu threshold foreground mask 
     (sitk.OtsuThreshold) on the fly.
  3. Smart Modality & Scan Discovery: Filters out segmentation masks (*seg.nii.gz) so N4 bias 
     correction is only applied to continuous MRI structural scans.
  4. Memory-Safe Sequential Processing: Explicit ITK object cleanup & garbage collection per patient.
  5. Automatic Resume & Skip Tracking: Skips completed subjects in 0.0s upon re-run.

Author: IDiA Lab
Date: August 2026
"""

import os
import sys
import time
import gc
import shutil
import argparse
import platform
import traceback
import importlib.metadata
import pandas as pd
import SimpleITK as sitk

# Try importing fpdf (fpdf2) for PDF log creation
try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

# ==========================================
# 1. USER CONFIGURATION & PRESETS
# ==========================================

# Pre-defined Datasets
DATASETS_CONFIG = {
    "1": {
        "name": "Brain-Mets-Lung",
        "input_root": r"/Users/rains/rainworkspace/2-IDIALab/Datasets/PKG_Brain-Mets-Lung-MRI-Path-Segs_radiology_images/Brain-Mets-Lung-MRI-Path-Segs",
        "output_root": r"/Users/rains/rainworkspace/2-IDIALab/Datasets/PKG_Brain-Mets-Lung-MRI-Path-Segs_radiology_images/Brain-Mets-Lung_MRI_BC_Output",
    },
    "2": {
        "name": "BraTS2025-MET",
        "input_root": r"/Users/rains/rainworkspace/2-IDIALab/Datasets/BraTS2026/MICCAI-LH-BraTS2025-MET-Challenge-Training",
        "output_root": r"/Users/rains/rainworkspace/2-IDIALab/Datasets/BraTS2026/BraTS2025-MET-Training-BC_complete",
    }
}

# N4 Algorithm Parameters
DEFAULT_PARAMETERS = {
    "shrink_factor": 2,          # Downsampling factor (1 = full res, 2 = 2x speedup default, 4 = fastest)
    "num_fitting_levels": 4,     # Multi-resolution fitting levels
    "num_iters": 50,             # Max iterations per fitting level
    "convergence_thresh": 1e-6,  # Convergence threshold
}

# Suffix for bias-corrected output files
OUTPUT_SUFFIX = "_BC.nii.gz"


# ==========================================
# 2. HELPER CLASS: PDF LOGGER
# ==========================================
class PDFLogger:
    """
    Custom logger that intercepts standard print statements and writes them simultaneously
    to the terminal console and to a dynamic PDF log file using FPDF.
    """
    def __init__(self, pdf_path):
        self.terminal = sys.stdout
        self.pdf_path = pdf_path
        self.log_content = []
        
    def write(self, message):
        self.terminal.write(message)
        if message:
            self.log_content.append(message)
            
    def flush(self):
        self.terminal.flush()
        
    def update_pdf(self):
        if not HAS_FPDF:
            return
            
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Courier", size=9)
        
        full_text = "".join(self.log_content)
        # Guard against non-latin-1 characters in PDF output
        full_text = full_text.encode('latin-1', 'replace').decode('latin-1')
        
        pdf.multi_cell(0, 4.5, text=full_text)
        
        try:
            pdf.output(self.pdf_path)
        except Exception as e:
            self.terminal.write(f"\n[Warning] Could not update PDF '{self.pdf_path}': {e}\n")


# ==========================================
# 3. BIAS CORRECTION CORE FUNCTION
# ==========================================
def run_bc_on_patient(patient_entry, shrink_factor=2, num_fitting_levels=4, num_iters=50, convergence_thresh=1e-6, copy_unprocessed=True):
    """
    Performs N4 Bias Field Correction on all structural scans of a patient.
    If an existing brain mask is found, it is used. Otherwise, generates an 
    automatic Otsu threshold foreground mask on the fly.
    """
    pid = patient_entry['id']
    structural_scans = patient_entry['structural_scans']
    mask_path = patient_entry.get('mask_path')
    out_dir = patient_entry['out_dir']
    other_files = patient_entry.get('other_files', [])
    
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"   Patient ID: {pid}")
    print(f"   Structural Scans to Correct ({len(structural_scans)}): {[os.path.basename(s) for s in structural_scans]}")
    if mask_path and os.path.exists(mask_path):
        print(f"   Brain Mask Source: Pre-existing file ({os.path.basename(mask_path)})")
    else:
        print("   Brain Mask Source: Automatic Otsu Thresholding (Foreground mask computed on the fly)")
        
    # Process each structural scan
    for img_path in structural_scans:
        scan_name = os.path.basename(img_path)
        
        # Derive output filename (e.g. t1ce_img.nii.gz -> t1ce_img_BC.nii.gz or t1c_BC.nii.gz)
        if scan_name.endswith(".nii.gz"):
            out_name = scan_name.replace(".nii.gz", OUTPUT_SUFFIX)
        elif scan_name.endswith(".nii"):
            out_name = scan_name.replace(".nii", OUTPUT_SUFFIX)
        else:
            out_name = scan_name + OUTPUT_SUFFIX
            
        out_bc_path = os.path.join(out_dir, out_name)
        
        print(f"   -> Processing: {scan_name} ...")
        img = sitk.ReadImage(img_path, sitk.sitkFloat32)
        
        # Determine mask to use
        if mask_path and os.path.exists(mask_path):
            mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)
        else:
            # Automatic Fallback: Otsu Thresholding to separate head/brain from background air
            mask = sitk.OtsuThreshold(img, 0, 1)
            
        # Optional Downsampling for speed
        if shrink_factor > 1:
            img_shrunk = sitk.Shrink(img, [shrink_factor] * img.GetDimension())
            mask_shrunk = sitk.Shrink(mask, [shrink_factor] * mask.GetDimension())
        else:
            img_shrunk, mask_shrunk = img, mask
            
        # Execute N4 Bias Field Correction
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([num_iters] * num_fitting_levels)
        corrector.SetConvergenceThreshold(convergence_thresh)
        corrector.Execute(img_shrunk, mask_shrunk)
        
        # Reconstruct full-resolution bias field and apply
        log_bias_field = corrector.GetLogBiasFieldAsImage(img)
        corrected_img = img / sitk.Exp(log_bias_field)
        
        sitk.WriteImage(corrected_img, out_bc_path)
        print(f"      [Saved] Bias-Corrected Scan: {out_name}")
        
        # Free memory
        del img, mask, img_shrunk, mask_shrunk, corrector, log_bias_field, corrected_img
        gc.collect()

    # Synchronize non-structural files (segmentation masks, etc.) as-is
    if copy_unprocessed and other_files:
        for other_p in other_files:
            bname = os.path.basename(other_p)
            dest_p = os.path.join(out_dir, bname)
            if not os.path.exists(dest_p):
                shutil.copy2(other_p, dest_p)
                print(f"   [Synchronized] Copied segmentation/label file as-is: {bname}")


# ==========================================
# 4. LOGGING & MAIN PIPELINE LOOP
# ==========================================
def logging_loop(operation_name, input_root, output_root, patient_entries, params, excel_root=None):
    """
    Main logging loop wrapping execution with PDF log and Excel error tracking.
    """
    os.makedirs(output_root, exist_ok=True)
    
    log_pdf_path = os.path.join(output_root, f"{operation_name}_pipeline_log.pdf")
    error_excel_path = os.path.join(output_root, f"{operation_name}_pipeline_errors.xlsx")
    
    pdf_logger = PDFLogger(log_pdf_path)
    sys.stdout = pdf_logger
    
    attempted_patients = []
    errors_list = []
    start_time = time.time()
    
    try:
        print("==================================================")
        print(f"OPERATION: {operation_name}")
        print("==================================================\n")
        
        print("--- ENVIRONMENT INFO ---")
        print(f"Python version: {platform.python_version()}")
        print(f"OS/Platform:   {platform.platform()}")
        print(f"SimpleITK:     {sitk.__version__}")
        print(f"Pandas:        {pd.__version__}")
        print(f"FPDF Available: {HAS_FPDF}")
        
        print("\n--- PATHS & CONFIGURATION ---")
        print(f"Input Root:   {input_root}")
        print(f"Output Root:  {output_root}")
        print(f"Log PDF:      {log_pdf_path}")
        print(f"Error Excel:  {error_excel_path}")
        
        print("\n--- PARAMETERS ---")
        for k, v in params.items():
            print(f"  {k}: {v}")
            
        print("\n--- PATIENT QUEUE ---")
        print(f"Total subjects to process: {len(patient_entries)}")
        for idx, p in enumerate(patient_entries):
            print(f"  [{idx+1}/{len(patient_entries)}] Patient ID: {p['id']}")
        print("--------------------------------------------------\n")
        
        pdf_logger.update_pdf()
        
        # Iterate over subjects
        skipped_count = 0
        processed_count = 0
        
        for idx, patient_entry in enumerate(patient_entries):
            pid = patient_entry['id']
            print(f"\n>> [{idx+1}/{len(patient_entries)}] Checking Subject: {pid} ...")
            
            # Check if all structural output scans already exist
            out_dir = patient_entry['out_dir']
            structural_scans = patient_entry['structural_scans']
            all_done = True
            for img_path in structural_scans:
                scan_name = os.path.basename(img_path)
                out_name = scan_name.replace(".nii.gz", OUTPUT_SUFFIX) if scan_name.endswith(".nii.gz") else scan_name + OUTPUT_SUFFIX
                out_bc_path = os.path.join(out_dir, out_name)
                if not (os.path.exists(out_bc_path) and os.path.getsize(out_bc_path) > 0):
                    all_done = False
                    break
                    
            if all_done:
                print(f"   [Skipped] All bias-corrected scans already exist in {out_dir}.")
                skipped_count += 1
                continue
                
            patient_start = time.time()
            attempted_patients.append(pid)
            
            try:
                run_bc_on_patient(
                    patient_entry,
                    shrink_factor=params['shrink_factor'],
                    num_fitting_levels=params['num_fitting_levels'],
                    num_iters=params['num_iters'],
                    convergence_thresh=params['convergence_thresh'],
                    copy_unprocessed=True
                )
                elapsed = time.time() - patient_start
                processed_count += 1
                print(f"   [Done] Successfully processed {pid} in {elapsed:.1f}s.")
                
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"   [ERROR] Failed to process patient {pid}: {error_msg}")
                traceback.print_exc(file=sys.stdout)
                errors_list.append({"Patient": pid, "Error": error_msg})
                
                try:
                    df_err = pd.DataFrame(errors_list)
                    df_err.to_excel(error_excel_path, index=False)
                except Exception as ex:
                    print(f"   [Warning] Could not update Excel error log: {ex}")
                    
            finally:
                pdf_logger.update_pdf()
                
    except Exception as e:
        print(f"\nCRITICAL PIPELINE ERROR: {e}")
        traceback.print_exc(file=sys.stdout)
        
    finally:
        total_time = time.time() - start_time
        print("\n==================================================")
        print("--- ENDING SUMMARY ---")
        print(f"Total subjects queued:    {len(patient_entries)}")
        print(f"Total subjects skipped:   {skipped_count}")
        print(f"Total subjects processed: {processed_count}")
        print(f"Total errors logged:      {len(errors_list)}")
        print(f"Total execution time:     {total_time:.1f}s")
        print("==================================================")
        
        try:
            pdf_logger.update_pdf()
            if errors_list:
                df_err = pd.DataFrame(errors_list)
                df_err.to_excel(error_excel_path, index=False)
        finally:
            sys.stdout = pdf_logger.terminal


# ==========================================
# 5. PATIENT DISCOVERY & ENTRYPOINT
# ==========================================
def discover_patient_files(input_root, output_root, patient_subset=None):
    """
    Scans the input root directory to discover patient folders and categorized files.
    Identifies structural images vs. segmentation masks vs. brain masks.
    """
    if not os.path.exists(input_root):
        print(f"FATAL: Input directory '{input_root}' does not exist.")
        sys.exit(1)
        
    all_entries = sorted(os.listdir(input_root))
    patient_ids = [d for d in all_entries if os.path.isdir(os.path.join(input_root, d)) and not d.startswith('.')]
    
    if patient_subset:
        try:
            start, end = map(int, patient_subset.split(':'))
            patient_ids = patient_ids[start:end]
            print(f"   [Subset Active] Selected patient range {patient_subset} ({len(patient_ids)} patients).")
        except Exception:
            print(f"   [Warning] Invalid PATIENT_SUBSET format '{patient_subset}'. Processing all patients.")
            
    patient_entries = []
    
    for pid in patient_ids:
        p_dir = os.path.join(input_root, pid)
        p_out_dir = os.path.join(output_root, pid)
        all_p_files = os.listdir(p_dir)
        
        # 1. Look for pre-existing brain mask
        mask_file = next((
            f for f in all_p_files 
            if ("bet" in f.lower() or "brain_mask" in f.lower()) and (f.endswith(".nii.gz") or f.endswith(".nii"))
        ), None)
        
        # 2. Look for segmentation masks (e.g. core_seg.nii.gz, whole_seg.nii.gz, -seg.nii.gz)
        seg_files = [
            f for f in all_p_files 
            if ("seg" in f.lower() or "label" in f.lower() or "tumor" in f.lower()) 
            and (f.endswith(".nii.gz") or f.endswith(".nii"))
            and f != mask_file
        ]
        
        # 3. Identify structural MRI scans (all other NIfTI images)
        structural_files = [
            f for f in all_p_files 
            if (f.endswith(".nii.gz") or f.endswith(".nii"))
            and f != mask_file
            and f not in seg_files
            and not f.startswith('.')
        ]
        
        if not structural_files:
            continue
            
        patient_entries.append({
            'id': pid,
            'structural_scans': [os.path.join(p_dir, f) for f in structural_files],
            'mask_path': os.path.join(p_dir, mask_file) if mask_file else None,
            'other_files': [os.path.join(p_dir, sf) for sf in seg_files],
            'out_dir': p_out_dir
        })
        
    return patient_entries


def main():
    parser = argparse.ArgumentParser(description="IDiA Lab Step 3 N4 Bias Field Correction Script")
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="1",
        choices=["1", "2", "custom"],
        help="Pre-configured dataset choice: 1 (Brain-Mets-Lung), 2 (BraTS2025-MET), or custom"
    )
    parser.add_argument("-i", "--input_root", type=str, default=None, help="Custom input root path")
    parser.add_argument("-o", "--output_root", type=str, default=None, help="Custom output root path")
    parser.add_argument("-s", "--subset", type=str, default=None, help="Patient subset range (e.g. '0:5')")
    parser.add_argument("--shrink_factor", type=int, default=2, help="Shrink factor for N4 bias correction")
    
    args = parser.parse_args()
    
    # Resolve input and output paths
    if args.dataset in DATASETS_CONFIG and not args.input_root:
        cfg = DATASETS_CONFIG[args.dataset]
        input_root = cfg["input_root"]
        output_root = cfg["output_root"]
        dataset_name = cfg["name"]
    else:
        input_root = args.input_root if args.input_root else DATASETS_CONFIG["1"]["input_root"]
        output_root = args.output_root if args.output_root else DATASETS_CONFIG["1"]["output_root"]
        dataset_name = "Custom_Dataset"
        
    params = DEFAULT_PARAMETERS.copy()
    params["shrink_factor"] = args.shrink_factor
    
    print(f"\n==================================================")
    print(f"   IDiA Step 3 N4 Bias Correction ({dataset_name})")
    print(f"==================================================")
    
    patient_entries = discover_patient_files(input_root, output_root, patient_subset=args.subset)
    print(f"Discovered {len(patient_entries)} valid subject folders to process.")
    
    safe_op_name = f"Step3_BiasCorrection_{dataset_name}"
    logging_loop(
        operation_name=safe_op_name,
        input_root=input_root,
        output_root=output_root,
        patient_entries=patient_entries,
        params=params
    )

if __name__ == "__main__":
    main()
