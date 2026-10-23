# -*- coding: utf-8 -*-
"""
IDiA Lab - Step 1 Rigid Co-registration Pipeline (Single-Worker Edition)
=======================================================================
File: rain_preprocess_step1_coregistration.py

Same logging / discovery / resume layout as rain_preprocess_step3_bc.py.
Registration itself is rain_preprocess.py Step 1: SimpleITK Mattes Mutual
Information, rigid Euler 3D, T1-CE / T1c as the fixed image.

Efficiency / no redundant outputs:
  * Output folder mirrors input: original filenames only (no _aligned suffix,
    no stacked 4D volumes, no extra multilabel maps).
  * If a moving volume already shares the T1-CE grid, it is copied — no MI, no resample.
  * Transforms are cached by grid, so a FLAIR-space segmentation reuses the FLAIR transform.
  * Per-file resume: existing non-empty outputs are skipped.

Usage:
  python rain_preprocess_step1_coregistration.py -d 1
  python rain_preprocess_step1_coregistration.py -d 2 -s 0:5

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
import numpy as np
import pandas as pd
import SimpleITK as sitk

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

# ==========================================
# 1. USER CONFIGURATION & PRESETS
# ==========================================

DATASETS_CONFIG = {
    "1": {
        "name": "Brain-Mets-Lung",
        "input_root": r"D:\A1_RainSun_20240916\1-UWMadison\IDiA-Lab\Medical_Images_Public\Brain-Mets-Lung-MRI-Path-Segs\PKG-Brain-Mets-Lung-MRI-Path-Segs_radiology_images\Brain-Mets-Lung-MRI-Path-Segs",
        "output_root": r"D:\A1_RainSun_20240916\1-UWMadison\IDiA-Lab\Medical_Images_Public\Brain-Mets-Lung-MRI-Path-Segs_Coregistered_Full",
    },
    "2": {
        "name": "BraTS2025-MET",
        "input_root": r"D:\A1_RainSun_20240916\1-UWMadison\IDiA-Lab\Medical_Images_Public\BraTS2026\MICCAI-LH-BraTS2025-MET-Challenge-Training",
        "output_root": r"D:\A1_RainSun_20240916\1-UWMadison\IDiA-Lab\Medical_Images_Public\BraTS2026\BraTS2025-MET-Training-Coregistered",
    }
}

DEFAULT_PARAMETERS = {
    "histogram_bins": 50,
    "sampling_percentage": 0.15,
    "num_iters": 100,
    "learning_rate": 1.0,
}

# ==========================================
# 2. HELPER CLASS: PDF LOGGER
# ==========================================
class PDFLogger:
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
        full_text = "".join(self.log_content).encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 4.5, text=full_text)
        try:
            pdf.output(self.pdf_path)
        except Exception as e:
            self.terminal.write(f"\n[Warning] Could not update PDF '{self.pdf_path}': {e}\n")


# ==========================================
# 3. REGISTRATION HELPERS
# ==========================================
def is_t1ce_reference(filename):
    """Contrast-enhanced T1 (fixed image). Does not match t1n / t1pre / t1w."""
    n = filename.lower()
    if not (n.endswith(".nii.gz") or n.endswith(".nii")):
        return False
    if "t1ce" in n or "t1post" in n or "t1-ce" in n or "t1_ce" in n:
        return True
    stem = n.replace(".nii.gz", "").replace(".nii", "")
    return stem.endswith("-t1c") or stem.endswith("_t1c") or "-t1c-" in stem


def is_label_map(filename, mask_file=None):
    n = filename.lower()
    if filename == mask_file:
        return True
    return any(k in n for k in ("seg", "label", "tumor", "mask", "bet"))


def grids_match(a, b, atol=1e-4):
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol)
    )


def grid_key(img):
    return (
        img.GetSize(),
        tuple(np.round(img.GetSpacing(), 5)),
        tuple(np.round(img.GetOrigin(), 4)),
        tuple(np.round(img.GetDirection(), 6)),
    )


def output_exists(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def copy_or_skip(src, dest, label):
    if output_exists(dest):
        print(f"      [Skipped] Already exists: {os.path.basename(dest)}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(src, dest)
    print(f"      [Copied] {label}: {os.path.basename(dest)}")


def register_rigid_mattes(fixed, moving, params):
    """rain_preprocess.py Step 1: Mattes MI rigid Euler 3D."""
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)
    tx = sitk.CenteredTransformInitializer(
        fixed_f,
        moving_f,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=params["histogram_bins"])
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(params["sampling_percentage"])
    R.SetInterpolator(sitk.sitkLinear)
    R.SetInitialTransform(tx, inPlace=True)
    R.SetOptimizerAsGradientDescent(
        learningRate=params["learning_rate"],
        numberOfIterations=params["num_iters"],
        estimateLearningRate=R.EachIteration,
    )
    R.SetOptimizerScalesFromPhysicalShift()
    return R.Execute(fixed_f, moving_f)


def resample_to_reference(moving, fixed, transform, is_label):
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    default_val = 0.0
    return sitk.Resample(moving, fixed, transform, interp, default_val, moving.GetPixelID())


# ==========================================
# 4. PER-PATIENT DRIVER
# ==========================================
def run_coreg_on_patient(patient_entry, params, copy_unprocessed=True):
    pid = patient_entry["id"]
    ref_path = patient_entry["ref_path"]
    moving_scans = patient_entry["moving_scans"]
    label_files = patient_entry.get("label_files", [])
    other_files = patient_entry.get("other_files", [])
    out_dir = patient_entry["out_dir"]

    os.makedirs(out_dir, exist_ok=True)

    print(f"   Patient ID: {pid}")
    print(f"   Reference (fixed T1-CE): {os.path.basename(ref_path)}")
    print(f"   Moving structural scans ({len(moving_scans)}): {[os.path.basename(s) for s in moving_scans]}")
    print(f"   Label / mask maps ({len(label_files)}): {[os.path.basename(s) for s in label_files]}")

    ref_img = sitk.ReadImage(ref_path)
    out_ref = os.path.join(out_dir, os.path.basename(ref_path))
    copy_or_skip(ref_path, out_ref, "reference (native T1-CE grid)")

    transform_by_grid = {grid_key(ref_img): sitk.Transform()}

    for img_path in moving_scans:
        scan_name = os.path.basename(img_path)
        dest = os.path.join(out_dir, scan_name)
        print(f"   -> Structural: {scan_name} ...")
        if output_exists(dest):
            print("      [Skipped] Already exists.")
            # Still cache transform if we can cheaply read the source grid
            moving = sitk.ReadImage(img_path)
            if grids_match(ref_img, moving):
                transform_by_grid[grid_key(moving)] = sitk.Transform()
            del moving
            continue

        t0 = time.time()
        moving = sitk.ReadImage(img_path)
        if grids_match(ref_img, moving):
            shutil.copy2(img_path, dest)
            transform_by_grid[grid_key(moving)] = sitk.Transform()
            print(f"      [Copied] Same grid as T1-CE ({time.time() - t0:.2f}s)")
        else:
            print("      [Coreg] Mattes MI rigid registration -> T1-CE ...")
            tx = register_rigid_mattes(ref_img, moving, params)
            aligned = resample_to_reference(moving, ref_img, tx, is_label=False)
            sitk.WriteImage(aligned, dest)
            transform_by_grid[grid_key(moving)] = tx
            print(f"      [Saved] {scan_name} ({time.time() - t0:.2f}s)  Euler: {list(tx.GetParameters())}")
            del aligned, tx
        del moving
        gc.collect()

    for lab_path in label_files:
        lab_name = os.path.basename(lab_path)
        dest = os.path.join(out_dir, lab_name)
        print(f"   -> Label: {lab_name} ...")
        if output_exists(dest):
            print("      [Skipped] Already exists.")
            continue
        t0 = time.time()
        lab = sitk.ReadImage(lab_path)
        key = grid_key(lab)
        if grids_match(ref_img, lab):
            shutil.copy2(lab_path, dest)
            print(f"      [Copied] Already on T1-CE grid ({time.time() - t0:.2f}s)")
        else:
            tx = transform_by_grid.get(key)
            if tx is None:
                tx = sitk.Transform()
                print("      [Resample] Unknown grid — header resample (identity) onto T1-CE.")
            else:
                print("      [Resample] Applying cached transform, nearest-neighbor.")
            aligned = resample_to_reference(lab, ref_img, tx, is_label=True)
            sitk.WriteImage(aligned, dest)
            print(f"      [Saved] {lab_name} ({time.time() - t0:.2f}s)")
            del aligned
        del lab
        gc.collect()

    if copy_unprocessed and other_files:
        for other_p in other_files:
            bname = os.path.basename(other_p)
            dest_p = os.path.join(out_dir, bname)
            if not output_exists(dest_p):
                shutil.copy2(other_p, dest_p)
                print(f"   [Synchronized] Copied extra file as-is: {bname}")

    del ref_img
    gc.collect()


# ==========================================
# 5. LOGGING & MAIN PIPELINE LOOP
# ==========================================
def expected_output_paths(patient_entry):
    out_dir = patient_entry["out_dir"]
    names = [os.path.basename(patient_entry["ref_path"])]
    names += [os.path.basename(p) for p in patient_entry["moving_scans"]]
    names += [os.path.basename(p) for p in patient_entry.get("label_files", [])]
    return [os.path.join(out_dir, n) for n in names]


def logging_loop(operation_name, input_root, output_root, patient_entries, params):
    os.makedirs(output_root, exist_ok=True)
    log_pdf_path = os.path.join(output_root, f"{operation_name}_pipeline_log.pdf")
    error_excel_path = os.path.join(output_root, f"{operation_name}_pipeline_errors.xlsx")
    pdf_logger = PDFLogger(log_pdf_path)
    sys.stdout = pdf_logger
    errors_list = []
    start_time = time.time()
    skipped_count = 0
    processed_count = 0

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

        for idx, patient_entry in enumerate(patient_entries):
            pid = patient_entry["id"]
            print(f"\n>> [{idx+1}/{len(patient_entries)}] Checking Subject: {pid} ...")
            if all(output_exists(p) for p in expected_output_paths(patient_entry)):
                print(f"   [Skipped] All co-registered files already exist in {patient_entry['out_dir']}.")
                skipped_count += 1
                continue
            try:
                t0 = time.time()
                run_coreg_on_patient(patient_entry, params=params, copy_unprocessed=True)
                processed_count += 1
                print(f"   [Done] Successfully processed {pid} in {time.time() - t0:.1f}s.")
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"   [ERROR] Failed to process patient {pid}: {error_msg}")
                traceback.print_exc(file=sys.stdout)
                errors_list.append({"Patient": pid, "Error": error_msg})
                try:
                    pd.DataFrame(errors_list).to_excel(error_excel_path, index=False)
                except Exception as ex:
                    print(f"   [Warning] Could not update Excel error log: {ex}")
            finally:
                pdf_logger.update_pdf()

    except Exception as e:
        print(f"\nCRITICAL PIPELINE ERROR: {e}")
        traceback.print_exc(file=sys.stdout)
    finally:
        print("\n==================================================")
        print("--- ENDING SUMMARY ---")
        print(f"Total subjects queued:    {len(patient_entries)}")
        print(f"Total subjects skipped:   {skipped_count}")
        print(f"Total subjects processed: {processed_count}")
        print(f"Total errors logged:      {len(errors_list)}")
        print(f"Total execution time:     {time.time() - start_time:.1f}s")
        print("==================================================")
        try:
            pdf_logger.update_pdf()
            if errors_list:
                pd.DataFrame(errors_list).to_excel(error_excel_path, index=False)
        finally:
            sys.stdout = pdf_logger.terminal


# ==========================================
# 6. PATIENT DISCOVERY & ENTRYPOINT
# ==========================================
def discover_patient_files(input_root, output_root, patient_subset=None):
    if not os.path.exists(input_root):
        print(f"FATAL: Input directory '{input_root}' does not exist.")
        sys.exit(1)

    all_entries = sorted(os.listdir(input_root))
    patient_ids = [d for d in all_entries if os.path.isdir(os.path.join(input_root, d)) and not d.startswith(".")]

    if patient_subset:
        try:
            start, end = map(int, patient_subset.split(":"))
            patient_ids = patient_ids[start:end]
            print(f"   [Subset Active] Selected patient range {patient_subset} ({len(patient_ids)} patients).")
        except Exception:
            print(f"   [Warning] Invalid PATIENT_SUBSET format '{patient_subset}'. Processing all patients.")

    patient_entries = []
    for pid in patient_ids:
        p_dir = os.path.join(input_root, pid)
        nifti = [f for f in os.listdir(p_dir) if f.endswith(".nii.gz") or f.endswith(".nii")]
        if not nifti:
            continue

        ref_file = next((f for f in nifti if is_t1ce_reference(f)), None)
        mask_file = next(
            (f for f in nifti if ("bet" in f.lower() or "brain_mask" in f.lower()) and f != ref_file),
            None,
        )
        label_files = [f for f in nifti if is_label_map(f, mask_file) and f != ref_file]
        if mask_file and mask_file not in label_files:
            label_files.append(mask_file)

        if not ref_file:
            structural = [f for f in nifti if f not in label_files]
            if not structural:
                continue
            ref_file = structural[0]

        moving = [f for f in nifti if f != ref_file and f not in label_files]
        extra = [f for f in os.listdir(p_dir) if f not in nifti and not f.startswith(".")]

        patient_entries.append({
            "id": pid,
            "ref_path": os.path.join(p_dir, ref_file),
            "moving_scans": [os.path.join(p_dir, f) for f in moving],
            "label_files": [os.path.join(p_dir, f) for f in label_files],
            "other_files": [os.path.join(p_dir, f) for f in extra],
            "out_dir": os.path.join(output_root, pid),
        })
    return patient_entries


def main():
    parser = argparse.ArgumentParser(description="IDiA Lab Step 1 Mattes MI Rigid Co-registration")
    parser.add_argument("-d", "--dataset", type=str, default="1", choices=["1", "2", "custom"],
                        help="1 = Brain-Mets-Lung CLEAN, 2 = BraTS2025-MET")
    parser.add_argument("-i", "--input_root", type=str, default=None)
    parser.add_argument("-o", "--output_root", type=str, default=None)
    parser.add_argument("-s", "--subset", type=str, default=None, help="Patient subset range (e.g. '0:5')")
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

    print("\n==================================================")
    print(f"   IDiA Step 1 Rigid Co-registration ({dataset_name})")
    print("==================================================")

    patient_entries = discover_patient_files(input_root, output_root, patient_subset=args.subset)
    print(f"Discovered {len(patient_entries)} valid subject folders to process.")

    logging_loop(
        operation_name=f"Step1_Coregistration_{dataset_name}",
        input_root=input_root,
        output_root=output_root,
        patient_entries=patient_entries,
        params=params,
    )


if __name__ == "__main__":
    main()
