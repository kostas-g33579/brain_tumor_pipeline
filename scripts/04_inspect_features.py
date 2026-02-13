"""
Step 4 — Feature Inspection & Quality Check
=============================================
Before running ComBat or classification, inspect the extracted feature matrix:
  1. Check feature completeness (NaN/Inf counts)
  2. Remove zero-variance features
  3. Check class balance
  4. Check site/batch distribution
  5. Generate summary statistics
  6. Produce a clean features CSV ready for ComBat / ML

Run this interactively or as a script after Step 3.

Usage:
    python 04_inspect_features.py --config ../config/config.yaml
"""

import os
import argparse
import yaml
import logging
import numpy as np
import pandas as pd
from datetime import datetime


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(cfg):
    os.makedirs(cfg["paths"]["logs_dir"], exist_ok=True)
    log_file = os.path.join(
        cfg["paths"]["logs_dir"],
        f"04_inspect_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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


def main():
    parser = argparse.ArgumentParser(description="Feature inspection and QC")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    logger = setup_logging(cfg)
    logger.info("Starting Step 4: Feature Inspection & QC")

    features_dir = cfg["paths"]["features_dir_pyradiomics"]
    combined_csv = os.path.join(features_dir, "features_combined.csv")

    if not os.path.exists(combined_csv):
        logger.error(f"Combined features CSV not found: {combined_csv}")
        logger.error("Run Step 3 first.")
        return

    df = pd.read_csv(combined_csv)
    logger.info(f"Loaded feature matrix: {df.shape[0]} cases × {df.shape[1]} columns")

    # ID/meta columns (not features)
    meta_cols = ["case_id", "label", "label_name", "dataset", "split", "site"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    logger.info(f"Meta columns: {len(meta_cols)} | Feature columns: {len(feat_cols)}")

    # ----------------------------------------------------------
    # 1. Class balance
    # ----------------------------------------------------------
    logger.info("\n--- CLASS BALANCE ---")
    for label_val, label_name in [(0, "Glioma"), (1, "Meningioma")]:
        n = len(df[df["label"] == label_val])
        logger.info(f"  {label_name} (label={label_val}): {n} cases ({100*n/len(df):.1f}%)")

    # ----------------------------------------------------------
    # 2. Site/batch distribution
    # ----------------------------------------------------------
    logger.info("\n--- SITE DISTRIBUTION ---")
    site_counts = df["site"].value_counts()
    for site, count in site_counts.head(10).items():
        logger.info(f"  Site {site}: {count} cases")
    if len(site_counts) > 10:
        logger.info(f"  ... and {len(site_counts)-10} more sites")
    logger.info(f"  Total unique sites/batches: {len(site_counts)}")

    # ----------------------------------------------------------
    # 3. Dataset split
    # ----------------------------------------------------------
    logger.info("\n--- SPLIT DISTRIBUTION ---")
    for split_val, cnt in df["split"].value_counts().items():
        logger.info(f"  {split_val}: {cnt} cases")

    # ----------------------------------------------------------
    # 4. Missing values
    # ----------------------------------------------------------
    logger.info("\n--- MISSING VALUES ---")
    feat_df = df[feat_cols]
    nan_counts = feat_df.isna().sum()
    nan_features = nan_counts[nan_counts > 0]
    inf_counts = np.isinf(feat_df.select_dtypes(include=np.number)).sum()
    inf_features = inf_counts[inf_counts > 0]

    logger.info(f"  Features with NaN values: {len(nan_features)}")
    logger.info(f"  Features with Inf values: {len(inf_features)}")

    if len(nan_features) > 0:
        logger.warning(f"  Top NaN features: {nan_features.head(5).to_dict()}")

    # ----------------------------------------------------------
    # 5. Zero/near-zero variance removal
    # ----------------------------------------------------------
    logger.info("\n--- VARIANCE FILTERING ---")
    numeric_feats = feat_df.select_dtypes(include=np.number)
    variances = numeric_feats.var()

    zero_var = variances[variances < 1e-8].index.tolist()
    low_var  = variances[(variances >= 1e-8) & (variances < 0.01)].index.tolist()

    logger.info(f"  Zero-variance features (will remove): {len(zero_var)}")
    logger.info(f"  Low-variance features  (< 0.01):       {len(low_var)}")
    logger.info(f"  Remaining features:                     {len(feat_cols) - len(zero_var)}")

    # ----------------------------------------------------------
    # 6. Correlation check (brief)
    # ----------------------------------------------------------
    logger.info("\n--- FEATURE STATISTICS (sample) ---")
    stats = numeric_feats.describe().T
    logger.info(f"  Mean of feature means:  {stats['mean'].mean():.4f}")
    logger.info(f"  Mean of feature stds:   {stats['std'].mean():.4f}")
    logger.info(f"  Features with mean > 100 (possibly un-normalized): "
                f"{len(stats[stats['mean'].abs() > 100])}")

    # ----------------------------------------------------------
    # 7. Clean and save
    # ----------------------------------------------------------
    logger.info("\n--- CLEANING ---")

    # Replace Inf with NaN
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)

    # Drop zero-variance features
    df_clean = df.drop(columns=zero_var, errors="ignore")
    logger.info(f"  Dropped {len(zero_var)} zero-variance features")

    # Drop features with >20% NaN
    feat_cols_clean = [c for c in feat_cols if c not in zero_var]
    nan_thresh = 0.20 * len(df_clean)
    high_nan_feats = [
        c for c in feat_cols_clean
        if df_clean[c].isna().sum() > nan_thresh
    ]
    df_clean = df_clean.drop(columns=high_nan_feats, errors="ignore")
    logger.info(f"  Dropped {len(high_nan_feats)} features with >20% NaN")

    # Fill remaining NaN with column median
    remaining_feat_cols = [c for c in df_clean.columns if c not in meta_cols]
    df_clean[remaining_feat_cols] = df_clean[remaining_feat_cols].fillna(
        df_clean[remaining_feat_cols].median()
    )
    remaining_nan = df_clean[remaining_feat_cols].isna().sum().sum()
    logger.info(f"  Filled remaining NaN with column median (remaining NaN: {remaining_nan})")

    # Save clean features
    clean_csv = os.path.join(features_dir, "features_clean.csv")
    df_clean.to_csv(clean_csv, index=False)
    logger.info(f"\n  Clean feature matrix saved: {clean_csv}")
    logger.info(f"  Final shape: {df_clean.shape[0]} cases × {df_clean.shape[1]} columns")

    # Save feature list for reference
    final_feat_cols = [c for c in df_clean.columns if c not in meta_cols]
    feat_list_path = os.path.join(features_dir, "feature_list.txt")
    with open(feat_list_path, "w") as f:
        for feat in final_feat_cols:
            f.write(feat + "\n")
    logger.info(f"  Feature list saved: {feat_list_path}")
    logger.info("Step 4 complete.")


if __name__ == "__main__":
    main()
