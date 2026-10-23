# -*- coding: utf-8 -*-
"""
IDiA Lab - Step 3 N4 Bias Field Correction Pipeline (Multi-Worker Server Edition)
===================================================================================
File: rain_preprocess_step3_bc_multiworker.py

A high-performance, multi-worker processing pipeline designed for large multi-socket CPU 
servers (e.g. 192 CPUs / 96 physical cores).

Key Features:
  1. Multi-Worker Parallel Scheduling (--num_workers / -w):
     - Parallel ProcessPoolExecutor scaling across all available CPU cores.
     - Single-Threaded ITK Worker Isolation (sitk.SetGlobalDefaultNumberOfThreads(1)) to prevent 
       CPU oversubscription and thread thrashing across NUMA sockets.
  2. Scheduling Policy & Duplicate Prevention:
     - Uses atomic lock files (.processing.lock) per subject directory to guarantee zero worker collision.
  3. Post-Pass Data Integrity Check & Automatic Rescheduling:
     - Scans 100% of subject directories after worker pool completion.
     - If any subject was interrupted or uncompleted, automatically schedules retry passes until all data is verified complete.
  4. Automatic Otsu Thresholding Fallback:
     - Uses pre-computed brain masks if available; otherwise computes foreground Otsu masks on the fly.
  5. Process-Safe Master PDF & Excel Error Logging:
     - Master process aggregates worker results and updates PDF reports and Excel error logs cleanly.

Usage:
  python rain_preprocess_step3_bc_multiworker.py -d 1 -w 32
  python rain_preprocess_step3_bc_multiworker.py -d 2 -w 64
  python rain_preprocess_step3_bc_multiworker.py -i /path/to/input -o /path/to/output -w 96

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
        "input_root": r"D:\A1_RainSun_20240916\1-UWMadison\IDiA-Lab\Medical_Images_Public\BraTS2026\MICCAI-LH-BraTS2025-MET-Challenge-Training",
        "output_root": r"D:\A1_RainSun_20240916\1-UWMadison\IDiA-Lab\Medical_Images_Public\BraTS2026\BraTS2025-MET-Training-BC_complete",
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
    
    # Process each structural scan
    for img_path in structural_scans:
        scan_name = os.path.basename(img_path)
        
        if scan_name.endswith(".nii.gz"):
            out_name = scan_name.replace(".nii.gz", OUTPUT_SUFFIX)
        elif scan_name.endswith(".nii"):
            out_name = scan_name.replace(".nii", OUTPUT_SUFFIX)
        else:
            out_name = scan_name + OUTPUT_SUFFIX
            
        out_bc_path = os.path.join(out_dir, out_name)
        
        # Skip scan if already computed and non-empty
        if os.path.exists(out_bc_path) and os.path.getsize(out_bc_path) > 0:
            continue
            
        img = sitk.ReadImage(img_path, sitk.sitkFloat32)
        
        # Determine mask to use
        if mask_path and os.path.exists(mask_path):
            mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)
        else:
            # Automatic Fallback: Otsu Thresholding
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


# ==========================================
# 4. WORKER TASK FUNCTION (PROCESS POOL)
# ==========================================
def _process_single_patient_worker(patient_entry, params):
    """
    Worker function executed inside ProcessPoolExecutor.
    - Forces SimpleITK single-threaded execution to prevent CPU thrashing across server sockets.
    - Uses atomic file locking (.processing.lock) to guarantee zero duplicate work across workers.
    """
    pid = patient_entry['id']
    out_dir = patient_entry['out_dir']
    structural_scans = patient_entry['structural_scans']
    
    # Restrict worker to single-threaded SimpleITK execution for server core scaling
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    
    # 1. Verification Check: Are all structural output scans already complete?
    all_done = True
    for img_path in structural_scans:
        scan_name = os.path.basename(img_path)
        out_name = scan_name.replace(".nii.gz", OUTPUT_SUFFIX) if scan_name.endswith(".nii.gz") else scan_name + OUTPUT_SUFFIX
        out_bc_path = os.path.join(out_dir, out_name)
        if not (os.path.exists(out_bc_path) and os.path.getsize(out_bc_path) > 0):
            all_done = False
            break
            
    if all_done:
        return {"patient": pid, "status": "skipped", "elapsed": 0.0, "error": None}
        
    # 2. Scheduling Policy: Acquire atomic lock to prevent duplicate worker collision
    os.makedirs(out_dir, exist_ok=True)
    lock_file = os.path.join(out_dir, ".processing.lock")
    
    if os.path.exists(lock_file):
        # Ignore locks older than 30 minutes (stale process guard)
        if time.time() - os.path.getmtime(lock_file) < 1800:
            return {"patient": pid, "status": "skipped_locked", "elapsed": 0.0, "error": None}
            
    try:
        with open(lock_file, 'w') as f:
            f.write(f"pid={pid}, time={time.time()}")
    except Exception:
        pass
        
    start_time = time.time()
    try:
        run_bc_on_patient(
            patient_entry,
            shrink_factor=params['shrink_factor'],
            num_fitting_levels=params['num_fitting_levels'],
            num_iters=params['num_iters'],
            convergence_thresh=params['convergence_thresh'],
            copy_unprocessed=True
        )
        elapsed = time.time() - start_time
        return {"patient": pid, "status": "done", "elapsed": elapsed, "error": None}
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        return {"patient": pid, "status": "error", "elapsed": time.time() - start_time, "error": error_msg}
    finally:
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except Exception:
                pass


# ==========================================
# 5. LOGGING & MULTI-WORKER PIPELINE LOOP
# ==========================================
def logging_loop(operation_name, input_root, output_root, patient_entries, params, num_workers=16, max_retries=3):
    """
    Master logging loop with multi-worker process pool, atomic duplicate prevention,
    and automatic post-pass verification & rescheduling.
    """
    os.makedirs(output_root, exist_ok=True)
    
    log_pdf_path = os.path.join(output_root, f"{operation_name}_pipeline_log.pdf")
    error_excel_path = os.path.join(output_root, f"{operation_name}_pipeline_errors.xlsx")
    
    pdf_logger = PDFLogger(log_pdf_path)
    sys.stdout = pdf_logger
    
    errors_list = []
    start_time = time.time()
    
    try:
        print("==================================================")
        print(f"OPERATION: {operation_name} (Multi-Worker Server Edition)")
        print("==================================================\n")
        
        print("--- ENVIRONMENT & HARDWARE INFO ---")
        print(f"Python version:   {platform.python_version()}")
        print(f"OS/Platform:      {platform.platform()}")
        print(f"SimpleITK:        {sitk.__version__}")
        print(f"Pandas:           {pd.__version__}")
        print(f"Parallel Workers: {num_workers}")
        print(f"Max Reschedules:  {max_retries}")
        print(f"FPDF Available:   {HAS_FPDF}")
        
        print("\n--- PATHS & CONFIGURATION ---")
        print(f"Input Root:   {input_root}")
        print(f"Output Root:  {output_root}")
        print(f"Log PDF:      {log_pdf_path}")
        print(f"Error Excel:  {error_excel_path}")
        
        print("\n--- PARAMETERS ---")
        for k, v in params.items():
            print(f"  {k}: {v}")
            
        print("\n--- PATIENT QUEUE ---")
        print(f"Total subjects queued: {len(patient_entries)}")
        print("--------------------------------------------------\n")
        
        pdf_logger.update_pdf()
        
        # Rescheduling & Scheduling Policy Loop
        remaining_entries = patient_entries.copy()
        attempt = 0
        total_processed = 0
        total_skipped = 0
        
        while remaining_entries and attempt <= max_retries:
            attempt += 1
            if attempt > 1:
                print(f"\n==================================================")
                print(f"--- RESCHEDULING RETRY PASS #{attempt-1} FOR {len(remaining_entries)} UNFINISHED SUBJECTS ---")
                print(f"==================================================\n")
                
            if num_workers > 1:
                import concurrent.futures
                print(f"Spinning up Multi-Worker Pool with {num_workers} processes ...")
                sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
                
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                    future_to_entry = {
                        executor.submit(_process_single_patient_worker, entry, params): entry
                        for entry in remaining_entries
                    }
                    
                    completed_count = 0
                    for future in concurrent.futures.as_completed(future_to_entry):
                        entry = future_to_entry[future]
                        pid = entry['id']
                        completed_count += 1
                        
                        try:
                            res = future.result()
                            status = res['status']
                            elapsed = res['elapsed']
                            
                            if status == "done":
                                total_processed += 1
                                print(f"   [{completed_count}/{len(remaining_entries)}] [Done] Subject {pid} ({elapsed:.1f}s)")
                            elif status == "skipped":
                                total_skipped += 1
                                print(f"   [{completed_count}/{len(remaining_entries)}] [Skipped] Subject {pid} (Already complete)")
                            elif status == "skipped_locked":
                                print(f"   [{completed_count}/{len(remaining_entries)}] [Locked] Subject {pid} (In progress by another worker)")
                            else:
                                error_msg = res.get('error', 'Unknown Error')
                                print(f"   [{completed_count}/{len(remaining_entries)}] [ERROR] Subject {pid}: {error_msg}")
                                errors_list.append({"Patient": pid, "Error": error_msg})
                        except Exception as exc:
                            print(f"   [{completed_count}/{len(remaining_entries)}] [Worker Exception] Subject {pid}: {exc}")
                            errors_list.append({"Patient": pid, "Error": str(exc)})
                            
                        if completed_count % 10 == 0 or completed_count == len(remaining_entries):
                            pdf_logger.update_pdf()
            else:
                # Sequential Mode
                for idx, entry in enumerate(remaining_entries):
                    pid = entry['id']
                    res = _process_single_patient_worker(entry, params)
                    if res['status'] == "done":
                        total_processed += 1
                        print(f"   [{idx+1}/{len(remaining_entries)}] [Done] Subject {pid} ({res['elapsed']:.1f}s)")
                    elif res['status'] == "skipped":
                        total_skipped += 1
                        print(f"   [{idx+1}/{len(remaining_entries)}] [Skipped] Subject {pid} (Already complete)")
                    else:
                        print(f"   [{idx+1}/{len(remaining_entries)}] [ERROR] Subject {pid}: {res['error']}")
                        errors_list.append({"Patient": pid, "Error": res['error']})
                    pdf_logger.update_pdf()

            # --- POST-PASS COMPLETION VERIFICATION CHECK ---
            print("\n--- POST-PASS DATA INTEGRITY VERIFICATION ---")
            unverified_entries = []
            for entry in patient_entries:
                pid = entry['id']
                out_dir = entry['out_dir']
                all_done = True
                for img_path in entry['structural_scans']:
                    scan_name = os.path.basename(img_path)
                    out_name = scan_name.replace(".nii.gz", OUTPUT_SUFFIX) if scan_name.endswith(".nii.gz") else scan_name + OUTPUT_SUFFIX
                    out_bc_path = os.path.join(out_dir, out_name)
                    if not (os.path.exists(out_bc_path) and os.path.getsize(out_bc_path) > 0):
                        all_done = False
                        break
                if not all_done:
                    unverified_entries.append(entry)
                    
            if not unverified_entries:
                print(">>> VERIFICATION SUCCESS: 100% of dataset subjects have been verified complete!")
                remaining_entries = []
            else:
                print(f">>> VERIFICATION WARNING: {len(unverified_entries)} subject(s) are still missing or incomplete.")
                remaining_entries = unverified_entries
                
    except Exception as e:
        print(f"\nCRITICAL PIPELINE ERROR: {e}")
        traceback.print_exc(file=sys.stdout)
        
    finally:
        total_time = time.time() - start_time
        print("\n==================================================")
        print("--- ENDING SUMMARY ---")
        print(f"Total subjects queued:    {len(patient_entries)}")
        print(f"Total subjects skipped:   {total_skipped}")
        print(f"Total subjects processed: {total_processed}")
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
# 6. PATIENT DISCOVERY & ENTRYPOINT
# ==========================================
def discover_patient_files(input_root, output_root, patient_subset=None):
    """
    Scans input root to discover patient subfolders and categorize structural scans vs masks.
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
        
        mask_file = next((
            f for f in all_p_files 
            if ("bet" in f.lower() or "brain_mask" in f.lower()) and (f.endswith(".nii.gz") or f.endswith(".nii"))
        ), None)
        
        seg_files = [
            f for f in all_p_files 
            if ("seg" in f.lower() or "label" in f.lower() or "tumor" in f.lower()) 
            and (f.endswith(".nii.gz") or f.endswith(".nii"))
            and f != mask_file
        ]
        
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
    parser = argparse.ArgumentParser(description="IDiA Lab Step 3 N4 Bias Field Correction Pipeline (Multi-Worker Server Edition)")
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
    parser.add_argument("-w", "--num_workers", type=int, default=16, help="Number of parallel worker processes (default: 16, recommended 32-96 on 192-CPU servers)")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum reschedule retry passes for missing files (default: 3)")
    parser.add_argument("--shrink_factor", type=int, default=2, help="Shrink factor for N4 bias correction")
    
    args = parser.parse_args()
    
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
    print(f"   IDiA Step 3 N4 Bias Correction ({dataset_name}) [Multi-Worker Server Edition]")
    print(f"==================================================")
    
    patient_entries = discover_patient_files(input_root, output_root, patient_subset=args.subset)
    print(f"Discovered {len(patient_entries)} valid subject folders to process.")
    
    safe_op_name = f"Step3_BiasCorrection_MultiWorker_{dataset_name}"
    logging_loop(
        operation_name=safe_op_name,
        input_root=input_root,
        output_root=output_root,
        patient_entries=patient_entries,
        params=params,
        num_workers=args.num_workers,
        max_retries=args.max_retries
    )

if __name__ == "__main__":
    main()
