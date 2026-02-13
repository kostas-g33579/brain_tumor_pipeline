"""
Step 3 — Radiomics Feature Extraction
======================================
Extracts PyRadiomics features from preprocessed NIfTI images.

For each case and each active modality:
  1. Load preprocessed (normalized) image
  2. Load segmentation mask → build whole-tumor ROI
  3. Run PyRadiomics extraction
  4. Save per-case results to CSV

Final output: one CSV per modality + a combined features CSV.

Usage:
    python 03_extract_features.py --config ../config/config.yaml
    python 03_extract_features.py --config ../config/config.yaml --case BraTS-GLI-00000-000
    python 03_extract_features.py --config ../config/config.yaml --n-jobs 8
"""

import os
import csv
import argparse
import logging
import yaml
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import radiomics
from radiomics import featureextractor
import SimpleITK as sitk


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(cfg):
    os.makedirs(cfg["paths"]["logs_dir"], exist_ok=True)
    log_file = os.path.join(
        cfg["paths"]["logs_dir"],
        f"03_extract_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=getattr(logging, cfg["logging"]["level"]),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if cfg["logging"]["save_logs"] else logging.NullHandler()
        ]
    )
    # Suppress verbose PyRadiomics logging unless in DEBUG mode
    rad_logger = logging.getLogger("radiomics")
    rad_logger.setLevel(logging.WARNING)
    return logging.getLogger(__name__)


def build_pyradiomics_params(cfg):
    """
    Build PyRadiomics parameter dictionary from config.
    """
    rad_cfg = cfg["radiomics"]
    params = {
        "setting": {
            "binWidth": rad_cfg["bin_width"],
            "resampledPixelSpacing": rad_cfg["resample_pixel_spacing"] if rad_cfg["resample"] else None,
            "interpolator": "sitkBSpline",
            "verbose": rad_cfg["verbose"],
        },
        "featureClass": {fc: [] for fc in rad_cfg["feature_classes"]}
    }
    # Remove None values
    params["setting"] = {k: v for k, v in params["setting"].items() if v is not None}
    return params


def build_roi_mask(seg_data, strategy, custom_labels=None):
    """
    Build binary ROI mask from segmentation.

    BraTS label convention:
        1 = NETC (non-enhancing tumor core / necrosis)
        2 = SNFH (surrounding FLAIR hyperintensity / edema)
        3 = ET   (enhancing tumor)

    whole_tumor = all three combined (labels 1+2+3)
    """
    if strategy == "whole_tumor":
        mask = (seg_data > 0).astype(np.uint8)
    elif strategy == "enhancing_only":
        mask = (seg_data == 3).astype(np.uint8)
    elif strategy == "custom" and custom_labels:
        mask = np.zeros_like(seg_data, dtype=np.uint8)
        for lbl in custom_labels:
            mask[seg_data == lbl] = 1
    else:
        mask = (seg_data > 0).astype(np.uint8)

    return mask


def nib_to_sitk(img_nib):
    """Convert nibabel image to SimpleITK image."""
    data   = img_nib.get_fdata(dtype=np.float32)
    affine = img_nib.affine
    spacing = np.abs(np.diag(affine)[:3]).tolist()

    sitk_img = sitk.GetImageFromArray(np.transpose(data, (2, 1, 0)))
    sitk_img.SetSpacing(spacing)
    return sitk_img


def extract_features_for_case(case_id, img_path, seg_path, extractor,
                               roi_strategy, custom_labels, min_roi_voxels):
    """
    Run PyRadiomics on a single case.
    Returns dict of features (with diagnostic_ keys removed) or None on failure.
    """
    try:
        # Load image
        img_nib = nib.load(img_path)
        img_sitk = nib_to_sitk(img_nib)

        # Load and build ROI mask
        seg_nib  = nib.load(seg_path)
        seg_data = seg_nib.get_fdata().astype(np.uint8)
        roi_mask = build_roi_mask(seg_data, roi_strategy, custom_labels)

        # Check minimum ROI size
        if np.sum(roi_mask) < min_roi_voxels:
            return None, f"ROI too small: {np.sum(roi_mask)} voxels"

        # Convert mask to SimpleITK
        mask_sitk = sitk.GetImageFromArray(np.transpose(roi_mask, (2, 1, 0)).astype(np.uint8))
        mask_sitk.CopyInformation(img_sitk)

        # Extract features
        result = extractor.execute(img_sitk, mask_sitk)

        # Filter to feature values only (remove diagnostics and metadata)
        features = {
            k: float(v) for k, v in result.items()
            if not k.startswith("diagnostics_") and not k.startswith("original_shape_")
        }
        # Keep shape features separately (they are modality-independent)
        shape_features = {
            k: float(v) for k, v in result.items()
            if k.startswith("original_shape_")
        }
        features.update(shape_features)

        return features, "ok"

    except Exception as e:
        return None, str(e)


def process_modality(modality, records, preproc_dir, features_dir,
                     cfg, extractor_params, logger):
    """
    Extract features for all cases for a single modality.
    """
    roi_cfg = cfg["segmentation"]
    rad_cfg = cfg["radiomics"]

    # Build extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()
    # Apply settings programmatically
    for key, val in extractor_params.get("setting", {}).items():
        extractor.settings[key] = val
    # Enable feature classes
    extractor.disableAllFeatures()
    for feature_class in extractor_params.get("featureClass", {}).keys():
        extractor.enableFeatureClassByName(feature_class)

    results = []
    ok = failed = skipped = 0

    for i, row in enumerate(records):
        case_id = row["case_id"]

        if i % 50 == 0:
            logger.info(f"  [{modality}] Progress: {i}/{len(records)}...")

        # Paths to preprocessed image and seg
        img_path = os.path.join(preproc_dir, case_id, f"{case_id}-{modality}_norm.nii.gz")
        seg_path = os.path.join(preproc_dir, case_id, f"{case_id}-seg.nii.gz")

        if not os.path.exists(img_path):
            logger.warning(f"  [{case_id}] Preprocessed {modality} not found, skipping")
            skipped += 1
            continue

        if not os.path.exists(seg_path):
            logger.warning(f"  [{case_id}] Segmentation mask not found, skipping")
            skipped += 1
            continue

        features, status = extract_features_for_case(
            case_id, img_path, seg_path,
            extractor,
            roi_cfg["roi_strategy"],
            roi_cfg.get("custom_labels", []),
            roi_cfg["min_roi_voxels"]
        )

        if features is None:
            logger.warning(f"  [{case_id}] {modality} extraction failed: {status}")
            failed += 1
            continue

        # Add identifying columns
        record = {
            "case_id":    case_id,
            "label":      row["label"],
            "label_name": row["label_name"],
            "dataset":    row["dataset"],
            "split":      row["split"],
            "site":       row["site"],
        }
        # Prefix feature names with modality
        for feat_name, feat_val in features.items():
            record[f"{modality}_{feat_name}"] = feat_val

        results.append(record)
        ok += 1

    logger.info(f"  [{modality}] Done: {ok} ok | {failed} failed | {skipped} skipped")
    return results


def main():
    parser = argparse.ArgumentParser(description="Feature extraction step")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--case",   default=None,  help="Single case for debugging")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Override n_jobs from config")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    logger = setup_logging(cfg)
    logger.info("Starting Step 3: Feature Extraction")

    paths      = cfg["paths"]
    active_mods = [k for k, v in cfg["modalities"].items() if v]
    logger.info(f"Active modalities: {active_mods}")

    preproc_dir  = os.path.join(paths["output_root"], "preprocessed")
    features_dir = paths["features_dir_pyradiomics"]
    os.makedirs(features_dir, exist_ok=True)

    # Load master index
    with open(paths["master_csv"], "r") as f:
        records = list(csv.DictReader(f))

    if args.case:
        records = [r for r in records if r["case_id"] == args.case]
        logger.info(f"Debug mode: single case {args.case}")

    # Filter to cases without missing files
    records = [r for r in records if r["has_missing"] == "no" and r["has_seg"] == "yes"]
    logger.info(f"Processing {len(records)} cases")

    # Build PyRadiomics params dict
    extractor_params = build_pyradiomics_params(cfg)

    # Extract features per modality
    all_mod_dfs = []
    for modality in active_mods:
        logger.info(f"Extracting features: modality = {modality}")
        results = process_modality(
            modality, records, preproc_dir, features_dir,
            cfg, extractor_params, logger
        )

        if not results:
            logger.warning(f"No results for modality {modality}")
            continue

        df = pd.DataFrame(results)

        # Save per-modality CSV
        mod_csv = os.path.join(features_dir, f"features_{modality}.csv")
        df.to_csv(mod_csv, index=False)
        logger.info(f"  Saved: {mod_csv} ({len(df)} cases, {len(df.columns)} columns)")

        all_mod_dfs.append(df)

    # Merge all modalities into combined feature matrix
    if len(all_mod_dfs) > 1:
        id_cols = ["case_id", "label", "label_name", "dataset", "split", "site"]

        # Start with first df, merge others on id columns
        combined = all_mod_dfs[0]
        for df in all_mod_dfs[1:]:
            # Drop duplicate id_cols from subsequent dfs before merging
            feat_cols = id_cols + [c for c in df.columns if c not in id_cols]
            combined = pd.merge(combined, df[feat_cols], on=id_cols, how="inner")

        combined_csv = os.path.join(features_dir, "features_combined.csv")
        combined.to_csv(combined_csv, index=False)
        logger.info(f"Combined feature matrix: {combined_csv}")
        logger.info(f"  Shape: {combined.shape[0]} cases × {combined.shape[1]} features")
    elif len(all_mod_dfs) == 1:
        # Single modality — just rename as combined too
        single_csv = os.path.join(features_dir, "features_combined.csv")
        all_mod_dfs[0].to_csv(single_csv, index=False)
        logger.info(f"Single modality output saved as combined: {single_csv}")

    logger.info("Step 3 complete.")


if __name__ == "__main__":
    main()
