#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IDiA Lab Dataset Structure Rearrangement Tool
============================================================
Converts raw medical imaging datasets (such as BraTS 2025/2026 MET Challenge or
Yale Brain Mets Lung) into standardized formats:

1. nnUNet Raw Dataset Format (nnUNet_raw/DatasetXXX_Name/imagesTr & labelsTr)
2. Unified Preprocessing Format (patient-based folders for rain_preprocess)

Author: IDiA Lab
"""

import os
import sys
import argparse
import glob
import shutil
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RearrangeDataset")


def find_brats_cases(input_dir):
    """
    Finds BraTS cases in input_dir (including top-level and subdirectories like 'UCSD - Training').
    Expected file patterns per case dir:
        - {CASE_ID}-t1c.nii.gz / .nii (T1CE)
        - {CASE_ID}-t1n.nii.gz / .nii (T1 native)
        - {CASE_ID}-t2f.nii.gz / .nii (T2 FLAIR)
        - {CASE_ID}-t2w.nii.gz / .nii (T2 weighted)
        - {CASE_ID}-seg.nii.gz / .nii (Segmentation mask)
    """
    cases = {}
    input_path = Path(input_dir)

    # Search all directories containing .nii or .nii.gz files
    for path in input_path.glob("**/*-t1c.nii*"):
        case_dir = path.parent
        case_id = case_dir.name

        # Avoid duplicates if already found
        if case_id in cases:
            continue

        # Find modalities
        t1c = list(case_dir.glob(f"{case_id}-t1c.nii*"))
        t1n = list(case_dir.glob(f"{case_id}-t1n.nii*"))
        t2f = list(case_dir.glob(f"{case_id}-t2f.nii*"))
        t2w = list(case_dir.glob(f"{case_id}-t2w.nii*"))
        seg = list(case_dir.glob(f"{case_id}-seg.nii*"))

        images = []
        if t1c: images.append((0, "T1C", str(t1c[0])))
        if t1n: images.append((1, "T1N", str(t1n[0])))
        if t2f: images.append((2, "T2F", str(t2f[0])))
        if t2w: images.append((3, "T2W", str(t2w[0])))

        label = str(seg[0]) if seg else None

        if images:
            cases[case_id] = {
                "case_id": case_id,
                "images": sorted(images, key=lambda x: x[0]),
                "label": label,
                "dataset_type": "brats"
            }

    return cases


def find_yale_cases(input_dir, yale_seg_type="whole"):
    """
    Finds Yale Brain-Mets-Lung cases in input_dir.
    Expected file patterns per case dir:
        - *_t1ce_img.nii.gz
        - *_flair_img.nii.gz
        - *_whole_seg.nii.gz or *_core_seg.nii.gz
    """
    cases = {}
    input_path = Path(input_dir)

    for path in input_path.glob("**/*_t1ce_img.nii*"):
        case_dir = path.parent
        case_id = case_dir.name

        if case_id in cases:
            continue

        t1ce = list(case_dir.glob("*_t1ce_img.nii*"))
        flair = list(case_dir.glob("*_flair_img.nii*"))
        whole_seg = list(case_dir.glob("*_whole_seg.nii*"))
        core_seg = list(case_dir.glob("*_core_seg.nii*"))

        images = []
        if t1ce: images.append((0, "T1CE", str(t1ce[0])))
        if flair: images.append((1, "FLAIR", str(flair[0])))

        label = None
        if yale_seg_type == "core" and core_seg:
            label = str(core_seg[0])
        elif whole_seg:
            label = str(whole_seg[0])
        elif core_seg:
            label = str(core_seg[0])

        if images:
            cases[case_id] = {
                "case_id": case_id,
                "images": sorted(images, key=lambda x: x[0]),
                "label": label,
                "dataset_type": "yale"
            }

    return cases


def auto_detect_and_find_cases(input_dir, dataset_type="auto", yale_seg_type="whole"):
    """
    Auto-detects dataset format or calls specific finder.
    """
    if dataset_type == "brats":
        return find_brats_cases(input_dir)
    elif dataset_type == "yale":
        return find_yale_cases(input_dir, yale_seg_type)
    else:
        # Auto detect
        brats_cases = find_brats_cases(input_dir)
        yale_cases = find_yale_cases(input_dir, yale_seg_type)

        if len(brats_cases) >= len(yale_cases) and len(brats_cases) > 0:
            logger.info(f"Auto-detected dataset type: BraTS (Found {len(brats_cases)} cases)")
            return brats_cases
        elif len(yale_cases) > 0:
            logger.info(f"Auto-detected dataset type: Yale Brain Mets (Found {len(yale_cases)} cases)")
            return yale_cases
        else:
            logger.warning("Auto-detection found 0 cases matching BraTS or Yale format!")
            return {}


def process_single_case(case_info, out_base_dir, target_format, file_action="copy", dry_run=False):
    """
    Processes a single case according to the target format (nnunet or patient_folders).
    """
    case_id = case_info["case_id"]
    images = case_info["images"]
    label = case_info["label"]

    actions_taken = []

    if target_format == "nnunet":
        images_tr_dir = os.path.join(out_base_dir, "imagesTr")
        labels_tr_dir = os.path.join(out_base_dir, "labelsTr")

        if not dry_run:
            os.makedirs(images_tr_dir, exist_ok=True)
            os.makedirs(labels_tr_dir, exist_ok=True)

        for channel_idx, channel_name, src_path in images:
            ext = ".nii.gz" if src_path.endswith(".nii.gz") else ".nii"
            dst_filename = f"{case_id}_{channel_idx:04d}{ext}"
            dst_path = os.path.join(images_tr_dir, dst_filename)
            actions_taken.append((src_path, dst_path))

        if label:
            ext = ".nii.gz" if label.endswith(".nii.gz") else ".nii"
            dst_filename = f"{case_id}{ext}"
            dst_path = os.path.join(labels_tr_dir, dst_filename)
            actions_taken.append((label, dst_path))

    elif target_format == "patient_folders":
        case_out_dir = os.path.join(out_base_dir, case_id)
        if not dry_run:
            os.makedirs(case_out_dir, exist_ok=True)

        for channel_idx, channel_name, src_path in images:
            ext = ".nii.gz" if src_path.endswith(".nii.gz") else ".nii"
            dst_filename = f"{channel_name.lower()}{ext}"
            dst_path = os.path.join(case_out_dir, dst_filename)
            actions_taken.append((src_path, dst_path))

        if label:
            ext = ".nii.gz" if label.endswith(".nii.gz") else ".nii"
            dst_filename = f"seg{ext}"
            dst_path = os.path.join(case_out_dir, dst_filename)
            actions_taken.append((label, dst_path))

    # Perform file transfer actions if not dry_run
    if not dry_run:
        for src, dst in actions_taken:
            if os.path.exists(dst):
                os.remove(dst)

            if file_action == "copy":
                shutil.copy2(src, dst)
            elif file_action == "symlink":
                os.symlink(os.path.abspath(src), dst)
            elif file_action == "move":
                shutil.move(src, dst)

    return case_id, len(actions_taken)


def create_dataset_json(out_base_dir, dataset_name, cases, channel_names, labels=None, dry_run=False):
    """
    Creates dataset.json required by nnUNet v2.
    """
    json_path = os.path.join(out_base_dir, "dataset.json")

    if labels is None:
        labels = {
            "background": 0,
            "tumor": 1
        }

    dataset_json_content = {
        "channel_names": channel_names,
        "labels": labels,
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "dataset_name": dataset_name
    }

    if not dry_run:
        with open(json_path, "w") as f:
            json.dump(dataset_json_content, f, indent=4)
        logger.info(f"Generated nnUNet dataset.json at: {json_path}")
    else:
        logger.info(f"[Dry-Run] Would generate dataset.json at: {json_path}")
        logger.info(f"Dataset JSON Preview: {json.dumps(dataset_json_content, indent=2)}")


def main():
    parser = argparse.ArgumentParser(
        description="Rearrange raw medical imaging datasets to nnUNet or rain_preprocess format."
    )
    parser.add_argument("-i", "--input-dir", required=True, help="Input directory containing raw patient dataset.")
    parser.add_argument("-o", "--output-dir", required=True, help="Output destination directory.")
    parser.add_argument("-t", "--dataset-type", choices=["auto", "brats", "yale"], default="auto",
                        help="Raw dataset type (auto, brats, or yale). Default: auto")
    parser.add_argument("-f", "--target-format", choices=["nnunet", "patient_folders"], default="nnunet",
                        help="Target directory format (nnunet or patient_folders). Default: nnunet")
    parser.add_argument("--dataset-id", type=int, default=100,
                        help="Dataset 3-digit ID for nnUNet (e.g. 100 -> Dataset100_...). Default: 100")
    parser.add_argument("--dataset-name", type=str, default="BrainMets",
                        help="Dataset name prefix for nnUNet. Default: BrainMets")
    parser.add_argument("--file-action", choices=["copy", "symlink", "move"], default="copy",
                        help="File operation mode: copy, symlink, or move. Default: copy")
    parser.add_argument("--yale-seg-type", choices=["whole", "core"], default="whole",
                        help="Segmentation type for Yale dataset (whole or core). Default: whole")
    parser.add_argument("-w", "--num-workers", type=int, default=8,
                        help="Number of parallel worker threads. Default: 8")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview case discovery and actions without transferring files.")

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    logger.info(f"Scanning dataset at: {input_dir}")
    cases = auto_detect_and_find_cases(input_dir, args.dataset_type, args.yale_seg_type)

    if not cases:
        logger.error("No valid cases discovered. Exiting.")
        sys.exit(1)

    logger.info(f"Discovered {len(cases)} cases.")

    # Determine output path
    if args.target_format == "nnunet":
        dataset_foldername = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
        out_base_dir = os.path.join(output_dir, dataset_foldername)
    else:
        out_base_dir = output_dir

    if not args.dry_run:
        os.makedirs(out_base_dir, exist_ok=True)

    logger.info(f"Target Format: {args.target_format} | Action: {args.file_action} | Output: {out_base_dir}")

    # Determine channel mapping for dataset.json
    sample_case = next(iter(cases.values()))
    channel_names = {str(c[0]): c[1] for c in sample_case["images"]}

    # Process cases in parallel
    successful = 0
    total_files = 0

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                process_single_case, case_info, out_base_dir, args.target_format, args.file_action, args.dry_run
            ): case_id
            for case_id, case_info in cases.items()
        }

        for future in as_completed(futures):
            case_id = futures[future]
            try:
                cid, file_count = future.result()
                successful += 1
                total_files += file_count
            except Exception as e:
                logger.error(f"Failed to process case {case_id}: {e}")

    logger.info(f"Successfully processed {successful}/{len(cases)} cases ({total_files} total files).")

    # Generate dataset.json if target format is nnunet
    if args.target_format == "nnunet":
        create_dataset_json(out_base_dir, args.dataset_name, cases, channel_names, dry_run=args.dry_run)

    if args.dry_run:
        logger.info("[Dry-Run Complete] No files were modified.")
    else:
        logger.info("Dataset rearrangement completed successfully.")


if __name__ == "__main__":
    main()
