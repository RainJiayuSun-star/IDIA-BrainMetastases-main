# -*- coding: utf-8 -*-
"""
IDiA Lab - Step 1 GPU-Accelerated Rigid Co-registration (rain_preprocess_step1_coregistration_gpu.py)
=====================================================================================================
GPU-accelerated rigid (6-DoF) co-registration of MRI modalities onto a T1-CE / T1c
reference grid. Uses PyTorch CUDA tensors, a multi-resolution pyramid, and sampled
differentiable Mutual Information (Mattes-style) — the GPU analog of
rain_preprocess.py's SimpleITK Mattes MI rigid registration.

Features:
  1. PyTorch CUDA GPU Acceleration: Euler 3D rigid alignment + grid_sample resampling.
  2. CPU Fallback: SimpleITK Mattes Mutual Information if CUDA is unavailable or --backend sitk.
  3. Multi-modal metric: sampled soft Mutual Information (T1-CE vs FLAIR / T2).
  4. Identity fast-path: if moving and fixed already share the same grid, skip optimization.
  5. Nearest-neighbor resampling for segmentation / label maps.
  6. PDF Logging & Excel Error Tracking, matching rain_preprocess_step3_bc_gpu.py.
  7. Automatic resume: skips subjects whose aligned outputs already exist.

Usage:
  python rain_preprocess_step1_coregistration_gpu.py -d 1
  python rain_preprocess_step1_coregistration_gpu.py -d 2 -s 0:10
  python rain_preprocess_step1_coregistration_gpu.py -d 1 --backend sitk

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

# ==========================================
# 1. USER CONFIGURATION & PRESETS
# ==========================================

DATASETS_CONFIG = {
    "1": {
        "name": "Brain-Mets-Lung",
        "input_root": r"D:\A1_RainSun_20240916\1-UWMadison\IDiA-Lab\Medical_Images_Public\Brain-Mets-Lung-MRI-Path-Segs_CLEAN",
        "output_root": r"D:\A1_RainSun_20240916\1-UWMadison\IDiA-Lab\Medical_Images_Public\Brain-Mets-Lung-MRI-Path-Segs_Coregistered_GPU",
    },
    "2": {
        "name": "BraTS2025-MET",
        "input_root": r"D:\A1_RainSun_20240916\1-UWMadison\IDiA-Lab\Medical_Images_Public\BraTS2026\MICCAI-LH-BraTS2025-MET-Challenge-Training",
        "output_root": r"D:\A1_RainSun_20240916\1-UWMadison\IDiA-Lab\Medical_Images_Public\BraTS2026\BraTS2025-MET-Training-Coregistered",
    }
}

DEFAULT_PARAMETERS = {
    "shrink_factor": 2,          # Coarsest pyramid downsample (GPU optimization)
    "num_iters": 80,             # Adam iterations per pyramid level
    "learning_rate": 0.05,       # Adam LR (angles in rad, translations in mm, internally scaled)
    "mi_bins": 32,               # Soft histogram bins for Mutual Information
    "sample_frac": 0.15,         # Fraction of foreground voxels sampled for MI (matches SimpleITK)
    "backend": "gpu",            # 'gpu' (PyTorch CUDA) or 'sitk' (SimpleITK CPU)
    "device": "cuda",
}

OUTPUT_SUFFIX = "_aligned.nii.gz"


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
# 3. FILE / GRID HELPERS
# ==========================================
def _aligned_name(filename):
    if filename.endswith(".nii.gz"):
        return filename[:-7] + OUTPUT_SUFFIX
    if filename.endswith(".nii"):
        return filename[:-4] + OUTPUT_SUFFIX
    return filename + OUTPUT_SUFFIX


def is_t1ce_reference(filename):
    """Identify contrast-enhanced T1 (fixed image). Avoid matching t1n / t1pre / t1w."""
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


def sitk_meta(img):
    sx, sy, sz = img.GetSize()  # SimpleITK (x, y, z)
    return {
        "origin": np.array(img.GetOrigin(), dtype=np.float32),
        "spacing": np.array(img.GetSpacing(), dtype=np.float32),
        "direction": np.array(img.GetDirection(), dtype=np.float32).reshape(3, 3),
        "shape": (sz, sy, sx),  # numpy / torch (D, H, W) = (z, y, x)
    }


# ==========================================
# 4. PYTORCH GPU RIGID REGISTRATION ENGINE
# ==========================================
def _euler_rotation_matrix(rx, ry, rz):
    """Build a 3x3 rotation matrix from Euler angles (radians, XYZ / ITK Euler3D order)."""
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    one = rx.new_ones(())
    zero = rx.new_zeros(())
    Rx = torch.stack([
        torch.stack([one, zero, zero]),
        torch.stack([zero, cx, -sx]),
        torch.stack([zero, sx, cx]),
    ])
    Ry = torch.stack([
        torch.stack([cy, zero, sy]),
        torch.stack([zero, one, zero]),
        torch.stack([-sy, zero, cy]),
    ])
    Rz = torch.stack([
        torch.stack([cz, -sz, zero]),
        torch.stack([sz, cz, zero]),
        torch.stack([zero, zero, one]),
    ])
    return Rz @ Ry @ Rx


def _physical_grid(meta, device):
    """Voxel centers of a SimpleITK image in physical XYZ (mm). Shape (D, H, W, 3)."""
    D, H, W = meta["shape"]
    zz = torch.arange(D, device=device, dtype=torch.float32)
    yy = torch.arange(H, device=device, dtype=torch.float32)
    xx = torch.arange(W, device=device, dtype=torch.float32)
    z, y, x = torch.meshgrid(zz, yy, xx, indexing="ij")
    idx_xyz = torch.stack([x, y, z], dim=-1)
    origin = torch.as_tensor(meta["origin"], device=device)
    spacing = torch.as_tensor(meta["spacing"], device=device)
    direction = torch.as_tensor(meta["direction"], device=device)
    return origin + (idx_xyz * spacing) @ direction.T


def _phys_to_grid_sample(phys_xyz, meta):
    """Convert physical XYZ (mm) to grid_sample coords in [-1, 1], last dim (x, y, z)."""
    D, H, W = meta["shape"]
    origin = torch.as_tensor(meta["origin"], device=phys_xyz.device)
    spacing = torch.as_tensor(meta["spacing"], device=phys_xyz.device)
    direction = torch.as_tensor(meta["direction"], device=phys_xyz.device)
    idx_xyz = ((phys_xyz - origin) @ torch.linalg.inv(direction).T) / spacing
    eps = 1e-6
    nx = 2.0 * idx_xyz[..., 0] / max(W - 1, 1) - 1.0
    ny = 2.0 * idx_xyz[..., 1] / max(H - 1, 1) - 1.0
    nz = 2.0 * idx_xyz[..., 2] / max(D - 1, 1) - 1.0
    return torch.stack([nx, ny, nz], dim=-1).clamp(-1.0 - eps, 1.0 + eps)


def _warp_volume(moving_t, params, center, moving_meta, phys_f, mode="bilinear"):
    """
    Resample moving onto the fixed grid.
    SimpleITK Euler3D: T(x) = R (x - C) + t + C
    Resampling uses the inverse: x_m = R^T (x_f - C - t) + C
    params: (rx, ry, rz, tx, ty, tz)
    """
    rx, ry, rz, tx, ty, tz = params
    R = _euler_rotation_matrix(rx, ry, rz)
    t = torch.stack([tx, ty, tz])
    phys_m = (phys_f - center - t) @ R + center
    grid = _phys_to_grid_sample(phys_m, moving_meta)
    warped = F.grid_sample(
        moving_t[None, None, ...],
        grid[None, ...],
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )[0, 0]
    return warped


def _mutual_information_loss(fixed, warped, mask, bins=32, sample_frac=0.15):
    """Negative sampled soft Mutual Information (differentiable Mattes-style)."""
    fg = mask > 0.5
    if int(fg.sum().item()) < bins:
        return warped.mean() * 0.0
    f = fixed[fg]
    w = warped[fg]
    n = f.numel()
    n_sample = max(bins, int(n * sample_frac))
    if n_sample < n:
        idx = torch.randint(0, n, (n_sample,), device=fixed.device)
        f = f[idx]
        w = w[idx]
    else:
        n_sample = n

    def _norm01(x):
        lo = torch.quantile(x.detach(), 0.01)
        hi = torch.quantile(x.detach(), 0.99)
        return ((x - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)

    f = _norm01(f)
    w = _norm01(w)
    centers = torch.linspace(0.0, 1.0, bins, device=fixed.device)
    sigma = 0.5 / bins
    pf = torch.exp(-0.5 * ((f[:, None] - centers[None, :]) / sigma) ** 2)
    pw = torch.exp(-0.5 * ((w[:, None] - centers[None, :]) / sigma) ** 2)
    pf = pf / (pf.sum(dim=1, keepdim=True) + 1e-8)
    pw = pw / (pw.sum(dim=1, keepdim=True) + 1e-8)
    joint = torch.einsum("nb,nc->bc", pf, pw) / float(n_sample)
    p_f = joint.sum(dim=1).clamp_min(1e-8)
    p_w = joint.sum(dim=0).clamp_min(1e-8)
    h_f = -(p_f * p_f.log()).sum()
    h_w = -(p_w * p_w.log()).sum()
    h_j = -(joint.clamp_min(1e-8) * joint.clamp_min(1e-8).log()).sum()
    return -(h_f + h_w - h_j)


def _sitk_centered_euler_params(fixed, moving, device):
    """Cheap CPU initializer (geometry), then hand parameters to the GPU optimizer."""
    tx = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    params = list(tx.GetParameters())  # rx, ry, rz, tx, ty, tz
    center = np.array(tx.GetFixedParameters(), dtype=np.float32)  # rotation center
    p = torch.tensor(params, dtype=torch.float32, device=device, requires_grad=True)
    c = torch.as_tensor(center, device=device, dtype=torch.float32)
    return p, c


def rigid_coregister_gpu(fixed_sitk, moving_sitk, params, device_str="cuda", is_label=False):
    """
    GPU rigid co-registration of moving onto fixed.
    Returns a SimpleITK image on the fixed grid with fixed metadata.
    """
    device = torch.device(device_str if torch.cuda.is_available() and device_str.startswith("cuda") else "cpu")
    shrink = int(params.get("shrink_factor", 2))
    n_iters = int(params.get("num_iters", 80))
    lr = float(params.get("learning_rate", 0.05))
    mi_bins = int(params.get("mi_bins", 32))
    sample_frac = float(params.get("sample_frac", 0.15))

    if grids_match(fixed_sitk, moving_sitk):
        aligned = sitk.Image(moving_sitk)
        aligned.CopyInformation(fixed_sitk)
        return aligned, "identity (grids already match)"

    fixed_f = sitk.Cast(fixed_sitk, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving_sitk, sitk.sitkFloat32)

    # Multi-resolution pyramid: coarse then fine (shrink, then 1)
    levels = []
    if shrink > 1:
        levels.append(shrink)
    levels.append(1)

    euler_params, center = _sitk_centered_euler_params(fixed_f, moving_f, device)

    for level_shrink in levels:
        if level_shrink > 1:
            fixed_lvl = sitk.Shrink(fixed_f, [level_shrink] * 3)
            moving_lvl = sitk.Shrink(moving_f, [level_shrink] * 3)
        else:
            # Fine level still slightly downsampled for speed if original shrink was 2
            fine_shrink = 2 if shrink >= 2 else 1
            if fine_shrink > 1:
                fixed_lvl = sitk.Shrink(fixed_f, [fine_shrink] * 3)
                moving_lvl = sitk.Shrink(moving_f, [fine_shrink] * 3)
            else:
                fixed_lvl, moving_lvl = fixed_f, moving_f

        fixed_meta = sitk_meta(fixed_lvl)
        moving_meta = sitk_meta(moving_lvl)
        fixed_t = torch.from_numpy(sitk.GetArrayFromImage(fixed_lvl).astype(np.float32)).to(device)
        moving_t = torch.from_numpy(sitk.GetArrayFromImage(moving_lvl).astype(np.float32)).to(device)
        mask_t = (fixed_t > torch.quantile(fixed_t, 0.10)).float()
        phys_f = _physical_grid(fixed_meta, device)

        opt = torch.optim.Adam([euler_params], lr=lr)
        level_iters = n_iters if level_shrink > 1 else max(20, n_iters // 2)

        for _ in range(level_iters):
            opt.zero_grad()
            warped = _warp_volume(moving_t, euler_params, center, moving_meta, phys_f, mode="bilinear")
            loss = _mutual_information_loss(fixed_t, warped, mask_t, bins=mi_bins, sample_frac=sample_frac)
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_([euler_params], 1.0)
            opt.step()

        del fixed_t, moving_t, mask_t, phys_f
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Full-resolution resample with the optimized Euler parameters
    with torch.no_grad():
        moving_full = torch.from_numpy(sitk.GetArrayFromImage(moving_f).astype(np.float32)).to(device)
        fixed_meta = sitk_meta(fixed_f)
        moving_meta = sitk_meta(moving_f)
        phys_f = _physical_grid(fixed_meta, device)
        mode = "nearest" if is_label else "bilinear"
        warped_full = _warp_volume(moving_full, euler_params, center, moving_meta, phys_f, mode=mode)
        out_arr = warped_full.detach().cpu().numpy().astype(np.float32 if not is_label else np.uint8)

    aligned = sitk.GetImageFromArray(out_arr)
    aligned.CopyInformation(fixed_sitk)
    p = euler_params.detach().cpu().numpy()
    summary = (
        f"Euler rx={p[0]:.4f} ry={p[1]:.4f} rz={p[2]:.4f} rad, "
        f"t=({p[3]:.2f},{p[4]:.2f},{p[5]:.2f}) mm"
    )
    del moving_full, warped_full, euler_params
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return aligned, summary


def rigid_coregister_sitk(fixed_sitk, moving_sitk, is_label=False):
    """CPU fallback matching rain_preprocess.py step_1_coregistration."""
    if grids_match(fixed_sitk, moving_sitk):
        aligned = sitk.Resample(moving_sitk, fixed_sitk, sitk.Transform(), sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear, 0.0, moving_sitk.GetPixelID())
        return aligned, "identity (grids already match)"

    fixed = sitk.Cast(fixed_sitk, sitk.sitkFloat32)
    moving = sitk.Cast(moving_sitk, sitk.sitkFloat32)
    tx = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.15)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetInitialTransform(tx, inPlace=True)
    R.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, estimateLearningRate=R.EachIteration)
    R.SetOptimizerScalesFromPhysicalShift()
    out_tx = R.Execute(fixed, moving)
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    aligned = sitk.Resample(moving_sitk, fixed_sitk, out_tx, interp, 0.0, moving_sitk.GetPixelID())
    return aligned, f"Euler parameters: {list(out_tx.GetParameters())}"


# ==========================================
# 5. PER-PATIENT DRIVER
# ==========================================
def run_coreg_on_patient(patient_entry, params, copy_unprocessed=True):
    pid = patient_entry["id"]
    ref_path = patient_entry["ref_path"]
    moving_scans = patient_entry["moving_scans"]
    seg_files = patient_entry.get("seg_files", [])
    other_files = patient_entry.get("other_files", [])
    out_dir = patient_entry["out_dir"]
    backend = params.get("backend", "gpu")
    device_str = params.get("device", "cuda")

    os.makedirs(out_dir, exist_ok=True)

    print(f"   Patient ID: {pid}")
    print(f"   Reference (fixed T1-CE): {os.path.basename(ref_path)}")
    print(f"   Moving structural scans ({len(moving_scans)}): {[os.path.basename(s) for s in moving_scans]}")
    print(f"   Label / seg maps ({len(seg_files)}): {[os.path.basename(s) for s in seg_files]}")

    ref_img = sitk.ReadImage(ref_path)
    out_ref = os.path.join(out_dir, _aligned_name(os.path.basename(ref_path)))
    if not (os.path.exists(out_ref) and os.path.getsize(out_ref) > 0):
        sitk.WriteImage(ref_img, out_ref)
        print(f"      [Saved] Reference on native grid: {os.path.basename(out_ref)}")
    else:
        print(f"      [Skipped] Reference already written: {os.path.basename(out_ref)}")

    def _align_one(src_path, is_label):
        scan_name = os.path.basename(src_path)
        out_path = os.path.join(out_dir, _aligned_name(scan_name))
        print(f"   -> Aligning ({backend.upper()}): {scan_name} ...")
        t0 = time.time()
        moving = sitk.ReadImage(src_path)
        if backend == "gpu" and torch.cuda.is_available():
            aligned, summary = rigid_coregister_gpu(ref_img, moving, params, device_str=device_str, is_label=is_label)
        else:
            aligned, summary = rigid_coregister_sitk(ref_img, moving, is_label=is_label)
        sitk.WriteImage(aligned, out_path)
        print(f"      [Saved] {os.path.basename(out_path)} ({time.time() - t0:.2f}s)  {summary}")
        del moving, aligned
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for img_path in moving_scans:
        _align_one(img_path, is_label=False)

    for seg_path in seg_files:
        _align_one(seg_path, is_label=True)

    if copy_unprocessed and other_files:
        for other_p in other_files:
            bname = os.path.basename(other_p)
            dest_p = os.path.join(out_dir, bname)
            if not os.path.exists(dest_p):
                shutil.copy2(other_p, dest_p)
                print(f"   [Synchronized] Copied extra file as-is: {bname}")

    del ref_img
    gc.collect()


# ==========================================
# 6. LOGGING LOOP
# ==========================================
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
        print(f"PyTorch:       {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Device:    {torch.cuda.get_device_name(0)}")
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

            expected = [_aligned_name(os.path.basename(patient_entry["ref_path"]))]
            expected += [_aligned_name(os.path.basename(p)) for p in patient_entry["moving_scans"]]
            expected += [_aligned_name(os.path.basename(p)) for p in patient_entry.get("seg_files", [])]
            all_done = True
            for name in expected:
                out_p = os.path.join(patient_entry["out_dir"], name)
                if not (os.path.exists(out_p) and os.path.getsize(out_p) > 0):
                    all_done = False
                    break
            if all_done:
                print(f"   [Skipped] All aligned scans already exist in {patient_entry['out_dir']}.")
                skipped_count += 1
                continue

            try:
                t0 = time.time()
                run_coreg_on_patient(patient_entry, params=params, copy_unprocessed=True)
                processed_count += 1
                print(f"   [Done] Successfully processed {pid} in {time.time() - t0:.2f}s.")
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
# 7. PATIENT DISCOVERY & ENTRYPOINT
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
        seg_files = [f for f in nifti if is_label_map(f, mask_file) and f != ref_file]
        if mask_file and mask_file not in seg_files:
            seg_files.append(mask_file)

        if not ref_file:
            structural = [f for f in nifti if f not in seg_files]
            if not structural:
                continue
            ref_file = structural[0]

        moving = [f for f in nifti if f != ref_file and f not in seg_files]
        extra = [f for f in os.listdir(p_dir) if f not in nifti and not f.startswith(".")]

        patient_entries.append({
            "id": pid,
            "ref_path": os.path.join(p_dir, ref_file),
            "moving_scans": [os.path.join(p_dir, f) for f in moving],
            "seg_files": [os.path.join(p_dir, f) for f in seg_files],
            "other_files": [os.path.join(p_dir, f) for f in extra],
            "out_dir": os.path.join(output_root, pid),
        })
    return patient_entries


def main():
    parser = argparse.ArgumentParser(description="IDiA Lab Step 1 GPU-Accelerated Rigid Co-registration")
    parser.add_argument("-d", "--dataset", type=str, default="1", choices=["1", "2", "custom"],
                        help="Pre-configured dataset: 1 (Brain-Mets-Lung CLEAN), 2 (BraTS2025-MET)")
    parser.add_argument("-i", "--input_root", type=str, default=None)
    parser.add_argument("-o", "--output_root", type=str, default=None)
    parser.add_argument("-s", "--subset", type=str, default=None, help="Patient subset range (e.g. '0:5')")
    parser.add_argument("--shrink_factor", type=int, default=2, help="Coarsest pyramid downsample factor")
    parser.add_argument("--backend", type=str, default="gpu", choices=["gpu", "sitk"])
    parser.add_argument("--device", type=str, default="cuda", help="PyTorch device (e.g. cuda, cuda:0)")
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
    params["backend"] = args.backend
    params["device"] = args.device

    if args.backend == "gpu" and not torch.cuda.is_available():
        print("[Warning] CUDA is not available on this system. Falling back to SimpleITK CPU backend.")
        params["backend"] = "sitk"

    print("\n==================================================")
    print(f"   IDiA Step 1 GPU Rigid Co-registration ({dataset_name})")
    print("==================================================")

    patient_entries = discover_patient_files(input_root, output_root, patient_subset=args.subset)
    print(f"Discovered {len(patient_entries)} valid subject folders to process.")

    logging_loop(
        operation_name=f"Step1_Coregistration_GPU_{dataset_name}",
        input_root=input_root,
        output_root=output_root,
        patient_entries=patient_entries,
        params=params,
    )


if __name__ == "__main__":
    main()
