"""
Step 2 — Preprocessing
======================
For each case in master_index.csv:
  1. Load active modality NIfTI files
  2. Validate image shape, affine, and voxel spacing
  3. Apply Z-score intensity normalization (brain-masked)
  4. Save normalized NIfTI files to output directory
  5. Flag any cases that fail for manual inspection

Note: BraTS data is already co-registered and resampled to 1mm³.
      GLI is already skull-stripped. MEN needs SynthStrip (handle separately).
      This script assumes skull-stripping is already done.

Usage:
    python 02_preprocess.py --config ../config/config.yaml
    python 02_preprocess.py --config ../config/config.yaml --case BraTS-GLI-00000-000
"""

import os
import csv
import argparse
import logging
import yaml
import numpy as np
import nibabel as nib
from pathlib import Path
from datetime import datetime


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(cfg):
    os.makedirs(cfg["paths"]["logs_dir"], exist_ok=True)
    log_file = os.path.join(
        cfg["paths"]["logs_dir"],
        f"02_preprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=getattr(logging, cfg["logging"]["level"]),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if cfg["logging"]["save_logs"] else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_master_index(csv_path):
    with open(csv_path, "r") as f:
        return list(csv.DictReader(f))


def zscore_normalize(img_data, mask=None):
    """
    Z-score normalize image intensities.
    If mask provided, compute mean/std from masked voxels only (non-zero brain region).
    """
    if mask is not None:
        brain_voxels = img_data[mask > 0]
    else:
        brain_voxels = img_data[img_data > 0]  # Non-zero voxels as proxy for brain

    if len(brain_voxels) == 0:
        return img_data  # Return as-is if no brain voxels found

    mean = np.mean(brain_voxels)
    std  = np.std(brain_voxels)

    if std < 1e-8:
        return np.zeros_like(img_data, dtype=np.float32)

    normalized = (img_data.astype(np.float32) - mean) / std
    # Zero out background (keep mask structure intact)
    if mask is not None:
        normalized[mask == 0] = 0.0
    return normalized


def validate_image(img, case_id, modality, logger):
    """
    Basic sanity checks on a loaded NIfTI image.
    Returns True if valid, False otherwise.
    """
    data = img.get_fdata()
    spacing = np.abs(np.diag(img.affine)[:3])

    # Check shape — BraTS standard is 240x240x155
    if data.ndim != 3:
        logger.error(f"  [{case_id}] {modality}: Expected 3D, got shape {data.shape}")
        return False

    # Check for NaN or Inf values
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        logger.warning(f"  [{case_id}] {modality}: Contains NaN or Inf values")
        return False

    # Check image is not empty
    if np.max(data) == 0:
        logger.error(f"  [{case_id}] {modality}: Image is entirely zero")
        return False

    # Check approximate voxel spacing (BraTS should be ~1mm³)
    for i, sp in enumerate(spacing):
        if sp < 0.5 or sp > 3.0:
            logger.warning(f"  [{case_id}] {modality}: Unusual voxel spacing axis {i}: {sp:.3f}mm")

    return True


def process_case(row, cfg, output_dir, active_modalities, logger):
    """
    Process a single case: load, validate, normalize, save.
    Returns: (case_id, status, message)
    """
    case_id = row["case_id"]
    results = {"case_id": case_id, "status": "ok", "notes": ""}

    case_out_dir = os.path.join(output_dir, case_id)
    os.makedirs(case_out_dir, exist_ok=True)

    # Load segmentation mask (needed for normalization ROI)
    seg_path = row["path_seg"]
    seg_data = None
    if seg_path and os.path.exists(seg_path):
        seg_img  = nib.load(seg_path)
        seg_data = seg_img.get_fdata().astype(np.uint8)
        # Create whole-brain-like mask: where seg > 0 OR surrounding brain
        # For normalization, use non-zero T1 region (loaded below)
    else:
        if row["has_seg"] == "yes":
            logger.warning(f"  [{case_id}] Seg mask path recorded but file missing: {seg_path}")
        results["notes"] += "no_seg;"

    # Process each active modality
    for mod, enabled in active_modalities.items():
        if not enabled:
            continue

        path_key = f"path_{mod}"
        img_path = row.get(path_key, "")

        if not img_path or not os.path.exists(img_path):
            msg = f"Missing file for {mod}: {img_path}"
            logger.warning(f"  [{case_id}] {msg}")
            results["notes"] += f"missing_{mod};"
            if not cfg["preprocessing"]["skip_missing"]:
                results["status"] = "failed"
                return results
            continue

        try:
            img  = nib.load(img_path)
            data = img.get_fdata(dtype=np.float32)

            # Validate
            if not validate_image(img, case_id, mod, logger):
                results["status"] = "warning"
                results["notes"] += f"invalid_{mod};"

            # Normalize
            if cfg["preprocessing"]["normalize"]:
                # Use non-zero voxels of this modality as brain mask proxy
                brain_mask = (data > 0).astype(np.float32)
                data = zscore_normalize(data, mask=brain_mask)

            # Save normalized image
            out_filename = f"{case_id}-{mod}_norm.nii.gz"
            out_path = os.path.join(case_out_dir, out_filename)
            nib.save(nib.Nifti1Image(data, img.affine, img.header), out_path)

            if cfg["logging"]["log_per_case"]:
                logger.debug(f"  [{case_id}] {mod}: saved → {out_path}")

        except Exception as e:
            logger.error(f"  [{case_id}] Error processing {mod}: {e}")
            results["status"] = "failed"
            results["notes"] += f"error_{mod};"

    # Copy segmentation mask to output dir (unchanged)
    if seg_path and os.path.exists(seg_path):
        import shutil
        seg_out = os.path.join(case_out_dir, f"{case_id}-seg.nii.gz")
        if not os.path.exists(seg_out):
            shutil.copy2(seg_path, seg_out)

    return results


def main():
    parser = argparse.ArgumentParser(description="Preprocessing step")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--case",   default=None,  help="Process single case only (for debugging)")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    logger = setup_logging(cfg)
    logger.info("Starting Step 2: Preprocessing")

    paths  = cfg["paths"]
    mods   = {k: v for k, v in cfg["modalities"].items() if v}  # active only
    logger.info(f"Active modalities: {list(mods.keys())}")

    # Output directory for preprocessed images
    preproc_dir = os.path.join(paths["output_root"], "preprocessed")
    os.makedirs(preproc_dir, exist_ok=True)

    # Load master index
    records = load_master_index(paths["master_csv"])
    logger.info(f"Loaded {len(records)} cases from master index")

    # Filter to single case if debugging
    if args.case:
        records = [r for r in records if r["case_id"] == args.case]
        if not records:
            logger.error(f"Case not found in master index: {args.case}")
            return
        logger.info(f"Debug mode: processing single case {args.case}")

    # Filter out cases with missing active modalities (if skip_missing=True)
    if cfg["preprocessing"]["skip_missing"]:
        before = len(records)
        records = [r for r in records if r["has_missing"] == "no"]
        skipped = before - len(records)
        if skipped > 0:
            logger.warning(f"Skipping {skipped} cases with missing modality files")

    # Process all cases
    results_log = []
    ok = failed = warned = 0

    for i, row in enumerate(records):
        if i % 100 == 0:
            logger.info(f"Progress: {i}/{len(records)} cases processed...")

        result = process_case(row, cfg, preproc_dir, mods, logger)
        results_log.append(result)

        if result["status"] == "ok":      ok += 1
        elif result["status"] == "warning": warned += 1
        else:                             failed += 1

    # Summary
    logger.info("=" * 55)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("=" * 55)
    logger.info(f"  Total processed: {len(records)}")
    logger.info(f"  OK:              {ok}")
    logger.info(f"  Warnings:        {warned}")
    logger.info(f"  Failed:          {failed}")
    logger.info("=" * 55)

    # Save preprocessing report
    report_path = os.path.join(paths["logs_dir"], "preprocessing_report.csv")
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "status", "notes"])
        writer.writeheader()
        writer.writerows(results_log)
    logger.info(f"Preprocessing report saved: {report_path}")
    logger.info("Step 2 complete.")


if __name__ == "__main__":
    main()
