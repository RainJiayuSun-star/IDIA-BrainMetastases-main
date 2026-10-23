# -*- coding: utf-8 -*-
"""
IDiA Lab - Step 3 GPU-Accelerated N4 Bias Field Correction Pipeline (rain_preprocess_step3_bc_gpu.py)
===================================================================================================
Reference algorithm is Jared V2 (logging_jbBiasCorrection_BrambleScript_V2.py):

  sitk.N4BiasFieldCorrectionImageFilter on sitk.Shrink'd volumes
  log_B = corrector.GetLogBiasFieldAsImage(full_res_image)
  I_corrected = image / sitk.Exp(log_B)   # background is NOT zeroed

For model-benchmark inference, use --backend sitk (numerically the Jared/ANTs N4).
--backend gpu is a CUDA N3/N4-style approximation (B-spline replaced by masked Gaussian)
on the same shrink grid and the same apply formula. Do not mix GPU-BC and sitk-BC
volumes in one benchmark table.

Backends:
  --backend sitk  SimpleITK N4ITK (Jared V2; required for benchmark-comparable output)
  --backend gpu   PyTorch CUDA N3/N4-style (default; same I/O + apply formula)
  --backend n4ax  n4ax if importable, else gpu

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

import torch
import torch.nn.functional as F

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

try:
    import n4ax
    HAS_N4AX = True
except ImportError:
    HAS_N4AX = False

# ==========================================
# 1. USER CONFIGURATION & PRESETS
# ==========================================

# Pre-defined Datasets
DATASETS_CONFIG = {
    "1": {
        "name": "Brain-Mets-Lung",
        "input_root": r"D:\A1_RainSun_20240916\1-UWMadison\IDiA-Lab\Medical_Images_Public\Brain-Mets-Lung-MRI-Path-Segs_Coregistered",
        "output_root": r"D:\A1_RainSun_20240916\1-UWMadison\IDiA-Lab\Medical_Images_Public\Brain-Mets-Lung-MRI-Path-Segs_Coregistered_BC_gpu",
    },
    "2": {
        "name": "BraTS2025-MET",
        "input_root": r"D:\A1_RainSun_20240916\1-UWMadison\IDiA-Lab\Medical_Images_Public\BraTS2026\MICCAI-LH-BraTS2025-MET-Challenge-Training",
        "output_root": r"D:\A1_RainSun_20240916\1-UWMadison\IDiA-Lab\Medical_Images_Public\BraTS2026\BraTS2025-MET-Training-BC_gpu",
    }
}

# N4 Algorithm Default Parameters (Jared V2)
DEFAULT_PARAMETERS = {
    "shrink_factor": 2,
    "num_fitting_levels": 4,
    "num_iters": 50,             # Max iterations per fitting level (Jared V2)
    "convergence_thresh": 1e-6,
    "backend": "gpu",            # 'gpu' | 'n4ax' | 'sitk'
    "device": "cuda",
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
# 3. N4 ENGINES (Jared V2 sitk + GPU-style)
# ==========================================
def _is_nifti(name):
    n = name.lower()
    return n.endswith(".nii.gz") or n.endswith(".nii")


def _is_brain_mask_file(name):
    n = name.lower()
    if not _is_nifti(name):
        return False
    return (
        "brain_mask" in n
        or n.endswith("_ss_bet.nii.gz")
        or n.endswith("_ss_bet.nii")
        or n.endswith("_bet.nii.gz")
        or n.endswith("_bet.nii")
        or "_ss_bet." in n
    )


def _is_seg_file(name):
    n = name.lower()
    if not _is_nifti(name) or _is_brain_mask_file(name):
        return False
    return any(k in n for k in ("seg", "label", "tumor"))


def _bc_output_name(scan_name):
    if scan_name.endswith(".nii.gz"):
        return scan_name[:-7] + OUTPUT_SUFFIX
    if scan_name.endswith(".nii"):
        return scan_name[:-4] + OUTPUT_SUFFIX
    return scan_name + OUTPUT_SUFFIX


def _paired_mask_name(img_file, all_files):
    """Jared V2: <stem>_SS.nii.gz -> <stem>_SS_bet.nii.gz; also stem_bet / brain_mask."""
    file_set = set(all_files)
    if img_file.endswith("_SS.nii.gz"):
        jared = img_file.replace("_SS.nii.gz", "_SS_bet.nii.gz")
        if jared in file_set:
            return jared
    if img_file.endswith("_SS.nii"):
        jared = img_file.replace("_SS.nii", "_SS_bet.nii")
        if jared in file_set:
            return jared
    stem = img_file[:-7] if img_file.endswith(".nii.gz") else (img_file[:-4] if img_file.endswith(".nii") else img_file)
    for cand in (
        stem + "_bet.nii.gz",
        stem + "_bet.nii",
        stem + "_brain_mask.nii.gz",
        stem + "_brain_mask.nii",
        stem + "_SS_bet.nii.gz",
    ):
        if cand in file_set:
            return cand
    return None


def _ensure_mask_on_image(mask_sitk, img_sitk):
    same = (
        mask_sitk.GetSize() == img_sitk.GetSize()
        and mask_sitk.GetSpacing() == img_sitk.GetSpacing()
        and mask_sitk.GetOrigin() == img_sitk.GetOrigin()
        and mask_sitk.GetDirection() == img_sitk.GetDirection()
    )
    if same:
        return sitk.Cast(mask_sitk, sitk.sitkUInt8)
    return sitk.Resample(
        mask_sitk, img_sitk, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
    )


def n4_bias_field_correction_sitk(
    img_sitk,
    mask_sitk,
    shrink_factor=2,
    num_fitting_levels=4,
    num_iters=50,
    convergence_thresh=1e-6,
):
    """Exact Jared V2 / rain_preprocess_step3_bc.py SimpleITK N4."""
    mask_sitk = _ensure_mask_on_image(mask_sitk, img_sitk)
    if shrink_factor > 1:
        img_shrunk = sitk.Shrink(img_sitk, [shrink_factor] * img_sitk.GetDimension())
        mask_shrunk = sitk.Shrink(mask_sitk, [shrink_factor] * img_sitk.GetDimension())
    else:
        img_shrunk, mask_shrunk = img_sitk, mask_sitk

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([int(num_iters)] * int(num_fitting_levels))
    corrector.SetConvergenceThreshold(float(convergence_thresh))
    corrector.Execute(img_shrunk, mask_shrunk)
    log_bias_field = corrector.GetLogBiasFieldAsImage(img_sitk)
    corrected_img = img_sitk / sitk.Exp(log_bias_field)
    del img_shrunk, mask_shrunk, corrector, log_bias_field
    return corrected_img


def _separable_gaussian(vol, mask, sigma, eps=1e-5):
    """Masked 3D separable Gaussian (normalized by smoothed mask so edges do not darken)."""
    if sigma < 0.5:
        return vol
    device = vol.device
    ksize = int(sigma * 3) | 1
    t = torch.arange(ksize, device=device, dtype=torch.float32) - (ksize // 2)
    k1d = torch.exp(-t ** 2 / (2.0 * sigma ** 2))
    k1d = (k1d / k1d.sum()).view(1, 1, -1)
    pad = ksize // 2
    num = vol * mask
    den = mask
    num = F.conv3d(num[None, None], k1d.view(1, 1, -1, 1, 1), padding=(pad, 0, 0))
    num = F.conv3d(num, k1d.view(1, 1, 1, -1, 1), padding=(0, pad, 0))
    num = F.conv3d(num, k1d.view(1, 1, 1, 1, -1), padding=(0, 0, pad))[0, 0]
    den = F.conv3d(den[None, None], k1d.view(1, 1, -1, 1, 1), padding=(pad, 0, 0))
    den = F.conv3d(den, k1d.view(1, 1, 1, -1, 1), padding=(0, pad, 0))
    den = F.conv3d(den, k1d.view(1, 1, 1, 1, -1), padding=(0, 0, pad))[0, 0]
    return num / torch.clamp(den, min=eps)


def _n3_histogram_sharpen(log_resid, mask, n_bins=200, fwhm=0.15):
    """
    N3-style 1D histogram sharpening (Sled 1998), used inside N4.
    Wiener-deconvolves the foreground log-intensity histogram, then maps voxels
    onto the sharpened distribution. No autograd needed.
    """
    fg = mask > 0.5
    if int(fg.sum().item()) < n_bins:
        return log_resid
    vals = log_resid[fg]
    vmin = torch.quantile(vals, 0.001)
    vmax = torch.quantile(vals, 0.999)
    if float((vmax - vmin).item()) < 1e-6:
        return log_resid
    vals_c = vals.clamp(vmin, vmax)
    hist = torch.histc(vals_c, bins=n_bins, min=float(vmin.item()), max=float(vmax.item())) + 1e-6
    bin_width = (vmax - vmin) / n_bins
    sigma_bins = (fwhm / 2.355) / torch.clamp(bin_width, min=1e-6)
    freqs = torch.fft.rfftfreq(n_bins, device=log_resid.device)
    g = torch.exp(-0.5 * (2.0 * torch.pi * freqs * sigma_bins) ** 2)
    h_f = torch.fft.rfft(hist)
    h_sharp = torch.fft.irfft(h_f * g / (g * g + 0.01), n=n_bins).clamp_min(1e-8)
    cdf_src = torch.cumsum(hist, 0)
    cdf_src = cdf_src / cdf_src[-1]
    cdf_dst = torch.cumsum(h_sharp, 0)
    cdf_dst = cdf_dst / cdf_dst[-1]
    # Invert sharpened CDF: src_cdf(value) -> intensity on sharpened histogram
    x = (vals_c - vmin) / (vmax - vmin) * (n_bins - 1)
    x0 = x.long().clamp(0, n_bins - 2)
    a = (x - x0.float()).clamp(0.0, 1.0)
    cdf_v = cdf_src[x0] * (1.0 - a) + cdf_src[x0 + 1] * a
    # Searchsorted on CDF for inverse mapping
    idx = torch.searchsorted(cdf_dst, cdf_v).clamp(1, n_bins - 1)
    c0 = cdf_dst[idx - 1]
    c1 = cdf_dst[idx]
    w = (cdf_v - c0) / torch.clamp(c1 - c0, min=1e-8)
    sharp_vals = vmin + (idx.float() - 1.0 + w) / (n_bins - 1) * (vmax - vmin)
    out = log_resid.clone()
    out[fg] = sharp_vals
    return out


def n4_bias_field_correction_gpu(
    img_sitk,
    mask_sitk,
    shrink_factor=2,
    num_fitting_levels=4,
    num_iters=50,
    convergence_thresh=1e-6,
    device_str="cuda",
):
    """
    GPU N4-style correction on the Jared V2 shrink grid.

    Same as Jared for: sitk.Shrink, full-res apply I/exp(log_B), background preserved.
    Different from Jared: B-spline N4 replaced by N3 histogram sharpening + masked Gaussian.
    """
    device = torch.device(device_str if torch.cuda.is_available() and str(device_str).startswith("cuda") else "cpu")
    mask_sitk = _ensure_mask_on_image(mask_sitk, img_sitk)

    if shrink_factor > 1:
        img_shrunk = sitk.Shrink(img_sitk, [shrink_factor] * img_sitk.GetDimension())
        mask_shrunk = sitk.Shrink(mask_sitk, [shrink_factor] * img_sitk.GetDimension())
    else:
        img_shrunk, mask_shrunk = img_sitk, mask_sitk

    img_sub = torch.from_numpy(sitk.GetArrayFromImage(img_shrunk).astype(np.float32)).to(device)
    mask_sub = (torch.from_numpy(sitk.GetArrayFromImage(mask_shrunk).astype(np.float32)).to(device) > 0).float()

    eps = 1e-5
    log_img = torch.log(torch.clamp(img_sub, min=eps))
    bias = torch.zeros_like(log_img)

    n_levels = max(1, int(num_fitting_levels))
    sigmas = [max(2.0, 16.0 / (2 ** i)) for i in range(n_levels)]

    fg = mask_sub > 0.5
    for sig in sigmas:
        for _ in range(int(num_iters)):
            resid = log_img - bias
            sharpened = _n3_histogram_sharpen(resid, mask_sub)
            update = _separable_gaussian(resid - sharpened, mask_sub, sigma=sig)
            if int(fg.sum().item()) > 0:
                update = update - update[fg].mean()
                mean_abs = float(update[fg].abs().mean().item())
            else:
                mean_abs = 0.0
            bias = bias + update
            if mean_abs < convergence_thresh:
                break

    bias_sitk = sitk.GetImageFromArray(bias.detach().cpu().numpy().astype(np.float32))
    bias_sitk.CopyInformation(img_shrunk)
    log_bias_full = sitk.Resample(
        bias_sitk, img_sitk, sitk.Transform(), sitk.sitkBSpline, 0.0, sitk.sitkFloat32
    )
    corrected_sitk = img_sitk / sitk.Exp(log_bias_full)

    del img_sub, mask_sub, log_img, bias, img_shrunk, mask_shrunk, bias_sitk, log_bias_full
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return corrected_sitk


def n4_bias_field_correction_n4ax(img_sitk, mask_sitk, num_fitting_levels=4, num_iters=50):
    """Optional JAX n4ax path (closest GPU match to ITK N4)."""
    vol = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
    mask = sitk.GetArrayFromImage(mask_sitk) > 0
    iters = tuple([int(num_iters)] * int(num_fitting_levels))
    kwargs = dict(iters=iters, over_relax=1.0, conv_threshold=1.5e-3)
    try:
        corrected, _log_bias = n4ax.n4(vol, mask=mask, **kwargs)
    except TypeError:
        corrected, _log_bias = n4ax.n4(vol, **kwargs)
    out = sitk.GetImageFromArray(np.asarray(corrected, dtype=np.float32))
    out.CopyInformation(img_sitk)
    return out


# ==========================================
# 4. BIAS CORRECTION CORE FUNCTION
# ==========================================
def run_bc_on_patient(patient_entry, params, copy_unprocessed=True):
    """
    Performs N4 Bias Field Correction on all structural scans of a patient.
    Mask pairing follows Jared V2 when *_SS / *_SS_bet files exist.
    """
    pid = patient_entry['id']
    structural_scans = patient_entry['structural_scans']
    mask_path = patient_entry.get('mask_path')
    mask_by_scan = patient_entry.get('mask_by_scan') or {}
    out_dir = patient_entry['out_dir']
    other_files = patient_entry.get('other_files', [])
    
    backend = params.get('backend', 'gpu')
    shrink_factor = params.get('shrink_factor', 2)
    device_str = params.get('device', 'cuda')
    num_fitting_levels = params.get('num_fitting_levels', 4)
    num_iters = params.get('num_iters', 50)
    convergence_thresh = params.get('convergence_thresh', 1e-6)
    
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"   Patient ID: {pid}")
    print(f"   Structural Scans to Correct ({len(structural_scans)}): {[os.path.basename(s) for s in structural_scans]}")
        
    # Process each structural scan
    for img_path in structural_scans:
        scan_name = os.path.basename(img_path)
        out_name = _bc_output_name(scan_name)
        out_bc_path = os.path.join(out_dir, out_name)
        
        print(f"   -> Processing ({backend.upper()}): {scan_name} ...")
        scan_t0 = time.time()
        img = sitk.ReadImage(img_path, sitk.sitkFloat32)
        
        scan_mask_path = mask_by_scan.get(scan_name) or mask_path
        if scan_mask_path and os.path.exists(scan_mask_path):
            print(f"      Brain Mask: {os.path.basename(scan_mask_path)}")
            mask = sitk.ReadImage(scan_mask_path, sitk.sitkUInt8)
        else:
            print("      Brain Mask: Automatic Otsu Thresholding (no paired *_bet / brain_mask)")
            mask = sitk.OtsuThreshold(img, 0, 1)
            
        if backend == "n4ax" and HAS_N4AX:
            corrected_img = n4_bias_field_correction_n4ax(
                img, mask,
                num_fitting_levels=num_fitting_levels,
                num_iters=num_iters,
            )
        elif backend == "gpu" and torch.cuda.is_available():
            corrected_img = n4_bias_field_correction_gpu(
                img_sitk=img,
                mask_sitk=mask,
                shrink_factor=shrink_factor,
                num_fitting_levels=num_fitting_levels,
                num_iters=num_iters,
                convergence_thresh=convergence_thresh,
                device_str=device_str,
            )
        else:
            if backend == "gpu" and not torch.cuda.is_available():
                print("      [Warning] CUDA unavailable; using SimpleITK N4 (Jared V2).")
            corrected_img = n4_bias_field_correction_sitk(
                img, mask,
                shrink_factor=shrink_factor,
                num_fitting_levels=num_fitting_levels,
                num_iters=num_iters,
                convergence_thresh=convergence_thresh,
            )
            
        sitk.WriteImage(corrected_img, out_bc_path)
        scan_elapsed = time.time() - scan_t0
        print(f"      [Saved] Bias-Corrected Scan: {out_name} (in {scan_elapsed:.2f}s)")
        
        del img, mask, corrected_img
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if copy_unprocessed and other_files:
        for other_p in other_files:
            bname = os.path.basename(other_p)
            dest_p = os.path.join(out_dir, bname)
            if not os.path.exists(dest_p):
                shutil.copy2(other_p, dest_p)
                print(f"   [Synchronized] Copied segmentation/label/mask file as-is: {bname}")


# ==========================================
# 5. LOGGING & MAIN PIPELINE LOOP
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
        print(f"PyTorch:       {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Device:    {torch.cuda.get_device_name(0)}")
        print(f"n4ax Available: {HAS_N4AX}")
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
                out_bc_path = os.path.join(out_dir, _bc_output_name(scan_name))
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
                    params=params,
                    copy_unprocessed=True
                )
                elapsed = time.time() - patient_start
                processed_count += 1
                print(f"   [Done] Successfully processed {pid} in {elapsed:.2f}s.")
                
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
# 6. PATIENT DISCOVERY & ENTRYPOINT
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
        
        mask_files = [f for f in all_p_files if _is_brain_mask_file(f)]
        seg_files = [f for f in all_p_files if _is_seg_file(f)]
        structural_files = [
            f for f in all_p_files
            if _is_nifti(f)
            and f not in mask_files
            and f not in seg_files
            and not f.startswith('.')
            and "_BC.nii" not in f
        ]
        
        if not structural_files:
            continue

        mask_by_scan = {}
        for sf in structural_files:
            paired = _paired_mask_name(sf, all_p_files)
            if paired:
                mask_by_scan[sf] = os.path.join(p_dir, paired)
        
        fallback_mask = None
        if mask_files:
            fallback_mask = os.path.join(p_dir, sorted(mask_files)[0])
            
        patient_entries.append({
            'id': pid,
            'structural_scans': [os.path.join(p_dir, f) for f in structural_files],
            'mask_path': fallback_mask,
            'mask_by_scan': mask_by_scan,
            'other_files': [os.path.join(p_dir, sf) for sf in (seg_files + mask_files)],
            'out_dir': p_out_dir
        })
        
    return patient_entries


def main():
    parser = argparse.ArgumentParser(description="IDiA Lab Step 3 GPU-Accelerated N4 Bias Field Correction Script")
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
    parser.add_argument("--shrink_factor", type=int, default=2, help="Shrink factor (Jared V2 default 2)")
    parser.add_argument("--num_iters", type=int, default=50, help="Iterations per fitting level (Jared V2 default 50)")
    parser.add_argument("--num_fitting_levels", type=int, default=4, help="N4 multi-resolution levels (Jared V2 default 4)")
    parser.add_argument("--backend", type=str, default="gpu", choices=["gpu", "n4ax", "sitk"],
                        help="sitk = Jared V2 SimpleITK N4 (use this for model benchmark). gpu = CUDA N4-style approx. n4ax = JAX N4 if installed.")
    parser.add_argument("--device", type=str, default="cuda", help="PyTorch GPU device (e.g. 'cuda', 'cuda:0')")
    
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
    params["num_iters"] = args.num_iters
    params["num_fitting_levels"] = args.num_fitting_levels
    params["backend"] = args.backend
    params["device"] = args.device

    if args.backend == "n4ax" and not HAS_N4AX:
        print("[Warning] n4ax is not installed. Falling back to PyTorch GPU backend.")
        params["backend"] = "gpu"
    if params["backend"] == "gpu" and not torch.cuda.is_available():
        print("[Warning] CUDA is not available on this system. Falling back to SimpleITK N4 (Jared V2).")
        params["backend"] = "sitk"

    if params["backend"] == "gpu":
        print("[Benchmark] GPU backend approximates ITK N4 (Gaussian field, not B-spline).")
        print("            For Jared-identical model-benchmark inputs, rerun with --backend sitk.")
    elif params["backend"] == "sitk":
        print("[Benchmark] SimpleITK N4 backend matches Jared V2 (GetLogBiasFieldAsImage + I/exp(log_B)).")
        
    print(f"\n==================================================")
    print(f"   IDiA Step 3 GPU N4 Bias Correction ({dataset_name})")
    print(f"==================================================")
    
    patient_entries = discover_patient_files(input_root, output_root, patient_subset=args.subset)
    print(f"Discovered {len(patient_entries)} valid subject folders to process.")
    
    safe_op_name = f"Step3_BiasCorrection_GPU_{dataset_name}"
    logging_loop(
        operation_name=safe_op_name,
        input_root=input_root,
        output_root=output_root,
        patient_entries=patient_entries,
        params=params
    )

if __name__ == "__main__":
    main()
