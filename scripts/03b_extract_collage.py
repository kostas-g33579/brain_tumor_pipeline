"""
Step 3b — CoLlAGe Feature Extraction
=====================================
Extracts CoLlAGe radiomics features from preprocessed NIfTI images.

CoLlAGe (Co-occurrence of Local Anisotropic Gradient Orientations) captures
texture patterns based on gradient orientations within tumor ROIs.

This script is designed to run inside the Docker container:
    docker run -v C:/Users/kplom/Documents/thesis:/data radxtools/collageradiomics-pip python /data/brain_tumor_classification/scripts/03b_extract_collage.py --config /data/brain_tumor_classification/config/config.yaml

For each case and each active modality:
  1. Load preprocessed (normalized) image
  2. Load segmentation mask → build whole-tumor ROI
  3. Run CoLlAGe extraction
  4. Save per-case results to CSV

Final output: one CSV per modality + a combined features CSV.

Usage (from Windows PowerShell, outside container):
    docker run -v C:/Users/kplom/Documents/thesis:/data radxtools/collageradiomics-pip python /data/brain_tumor_classification/scripts/03b_extract_collage.py --config /data/brain_tumor_classification/config/config.yaml
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

from collageradiomics import Collage


def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Translate Windows paths to Docker mount paths
    # C:/Users/kplom/Documents/thesis → /data
    def translate_path(path):
        if path and isinstance(path, str):
            # Handle Windows paths
            if path.startswith("C:/Users/kplom/Documents/thesis"):
                return path.replace("C:/Users/kplom/Documents/thesis", "/data")
            # Handle already-translated paths
            elif path.startswith("/data"):
                return path
        return path
    
    # Translate all paths in config
    for key in cfg["paths"]:
        cfg["paths"][key] = translate_path(cfg["paths"][key])
    
    return cfg


def setup_logging(cfg):
    os.makedirs(cfg["paths"]["logs_dir"], exist_ok=True)
    log_file = os.path.join(
        cfg["paths"]["logs_dir"],
        f"03b_extract_collage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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


def build_roi_mask(seg_data):
    """
    Build binary ROI mask from segmentation (whole tumor = all non-zero labels).
    """
    return (seg_data > 0).astype(np.uint8)


def extract_collage_for_case(case_id, img_path, seg_path, svd_radius, num_unique_angles, min_roi_voxels):
    """
    Run CoLlAGe on a single case.
    
    Returns dict of aggregated Haralick features (mean, std) or None on failure.
    
    CoLlAGe produces voxel-wise feature maps for 13 Haralick features.
    We aggregate them using mean and std over the masked ROI.
    
    Features:
        0: Angular Second Moment (Energy)
        1: Contrast
        2: Correlation
        3: Sum of Squares (Variance)
        4: Inverse Difference Moment (Homogeneity)
        5: Sum Average
        6: Sum Variance
        7: Sum Entropy
        8: Entropy
        9: Difference Variance
        10: Difference Entropy
        11: Information Measure of Correlation 1
        12: Information Measure of Correlation 2
    """
    try:
        # Load image
        img_nib = nib.load(img_path)
        img_data = img_nib.get_fdata().astype(np.float32)
        
        # Load and build ROI mask
        seg_nib = nib.load(seg_path)
        seg_data = seg_nib.get_fdata().astype(np.uint8)
        roi_mask = build_roi_mask(seg_data)
        
        # Check minimum ROI size
        if np.sum(roi_mask) < min_roi_voxels:
            return None, f"ROI too small: {np.sum(roi_mask)} voxels"
        
        # Create CoLlAGe object
        collage = Collage(
            img_data,
            roi_mask,
            svd_radius=svd_radius,
            num_unique_angles=num_unique_angles
        )
        
        # Execute CoLlAGe computation
        collage.execute()
        
        # Feature names corresponding to indices 0-12
        feature_names = [
            "Energy", "Contrast", "Correlation", "Variance", "Homogeneity",
            "SumAverage", "SumVariance", "SumEntropy", "Entropy",
            "DifferenceVariance", "DifferenceEntropy",
            "InformationMeasureCorr1", "InformationMeasureCorr2"
        ]
        
        # Extract voxel-wise feature maps and aggregate
        # CoLlAGe output shape: (H, W, D, 13, 2) where 2 = primary (0) and secondary (1) angle
        # We use primary angle (index 0)
        features = {}
        
        for feat_idx, feat_name in enumerate(feature_names):
            # Get voxel-wise feature map for this feature (primary angle)
            feature_map = collage.get_single_feature_output(feat_idx)[:, :, :, 0]
            
            # Extract values only within the mask (ignore NaN outside ROI)
            masked_values = feature_map[roi_mask > 0]
            masked_values = masked_values[~np.isnan(masked_values)]
            
            if len(masked_values) > 0:
                # Aggregate: compute mean and std
                features[f"{feat_name}_mean"] = float(np.mean(masked_values))
                features[f"{feat_name}_std"] = float(np.std(masked_values))
            else:
                features[f"{feat_name}_mean"] = np.nan
                features[f"{feat_name}_std"] = np.nan
        
        return features, "ok"
    
    except Exception as e:
        return None, str(e)


def process_modality(modality, records, preproc_dir, collage_dir, cfg, logger):
    """
    Extract CoLlAGe features for all cases for a single modality.
    """
    collage_cfg = cfg["collage"]
    roi_cfg = cfg["segmentation"]
    
    svd_radius = collage_cfg.get("svd_radius", 5)
    num_unique_angles = collage_cfg.get("num_unique_angles", 64)
    min_roi_voxels = roi_cfg["min_roi_voxels"]
    
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
        
        features, status = extract_collage_for_case(
            case_id, img_path, seg_path,
            svd_radius, num_unique_angles, min_roi_voxels
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
            record[f"{modality}_collage_{feat_name}"] = feat_val
        
        results.append(record)
        ok += 1
    
    logger.info(f"  [{modality}] Done: {ok} ok | {failed} failed | {skipped} skipped")
    return results


def main():
    parser = argparse.ArgumentParser(description="CoLlAGe feature extraction step")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--case",   default=None,  help="Single case for debugging")
    args = parser.parse_args()
    
    cfg    = load_config(args.config)
    logger = setup_logging(cfg)
    logger.info("Starting Step 3b: CoLlAGe Feature Extraction")
    
    paths      = cfg["paths"]
    active_mods = [k for k, v in cfg["modalities"].items() if v]
    logger.info(f"Active modalities: {active_mods}")
    
    preproc_dir = os.path.join(paths["output_root"], "preprocessed")
    collage_dir = paths["features_dir_collage"]
    os.makedirs(collage_dir, exist_ok=True)
    
    # Load master index
    with open(paths["master_csv"], "r") as f:
        records = list(csv.DictReader(f))
    
    if args.case:
        records = [r for r in records if r["case_id"] == args.case]
        logger.info(f"Debug mode: single case {args.case}")
    
    # Filter to cases without missing files
    records = [r for r in records if r["has_missing"] == "no" and r["has_seg"] == "yes"]
    logger.info(f"Processing {len(records)} cases")
    
    # Extract features per modality
    all_mod_dfs = []
    for modality in active_mods:
        logger.info(f"Extracting CoLlAGe features: modality = {modality}")
        results = process_modality(
            modality, records, preproc_dir, collage_dir,
            cfg, logger
        )
        
        if not results:
            logger.warning(f"No results for modality {modality}")
            continue
        
        df = pd.DataFrame(results)
        
        # Save per-modality CSV
        mod_csv = os.path.join(collage_dir, f"collage_{modality}.csv")
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
        
        combined_csv = os.path.join(collage_dir, "collage_combined.csv")
        combined.to_csv(combined_csv, index=False)
        logger.info(f"Combined CoLlAGe feature matrix: {combined_csv}")
        logger.info(f"  Shape: {combined.shape[0]} cases × {combined.shape[1]} features")
    elif len(all_mod_dfs) == 1:
        # Single modality — just rename as combined too
        single_csv = os.path.join(collage_dir, "collage_combined.csv")
        all_mod_dfs[0].to_csv(single_csv, index=False)
        logger.info(f"Single modality output saved as combined: {single_csv}")
    
    logger.info("Step 3b complete.")


if __name__ == "__main__":
    main()
