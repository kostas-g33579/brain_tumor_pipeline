"""
Segmentation Comparison Study - Case Selection
================================================
Randomly selects a balanced subset of cases for comparing segmentation methods.
Copies selected cases to organized folder structure for comparison study.

Selects:
  - 25 glioma cases (from BraTS-GLI)
  - 25 meningioma cases (from BraTS-MEN)
  - All from training sets only
  - With existing segmentation masks

Creates folder structure:
  thesis_outputs/segmentation_comparison/
    ├── test_subset/GLI/  (copied cases)
    ├── test_subset/MEN/  (copied cases)
    ├── ground_truth_masks/  (original segmentation masks)
    └── segmentation_test_subset.csv

Usage:
    python 01_select_test_subset.py --config ../../config/config.yaml --n-per-class 25 --copy-data
"""

import os
import csv
import yaml
import random
import shutil
import argparse
from pathlib import Path


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def copy_case_data(case_id, dataset, data_dir, dest_dir, copy_masks_dir):
    """
    Copy all modalities and segmentation mask for a case.
    """
    case_path = os.path.join(data_dir, case_id)
    
    if not os.path.exists(case_path):
        print(f"Warning: Case directory not found: {case_path}")
        return False
    
    # Create destination directory
    dest_case_path = os.path.join(dest_dir, case_id)
    os.makedirs(dest_case_path, exist_ok=True)
    
    # Copy all .nii.gz files (modalities + segmentation)
    copied_files = []
    for file in os.listdir(case_path):
        if file.endswith(".nii.gz"):
            src = os.path.join(case_path, file)
            dst = os.path.join(dest_case_path, file)
            shutil.copy2(src, dst)
            copied_files.append(file)
            
            # Also copy ground truth mask to separate folder
            if file.endswith("-seg.nii.gz"):
                mask_dst = os.path.join(copy_masks_dir, file)
                shutil.copy2(src, mask_dst)
    
    print(f"  Copied {case_id}: {len(copied_files)} files")
    return True


def main():
    parser = argparse.ArgumentParser(description="Select subset for segmentation comparison")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--n-per-class", type=int, default=25, help="Number of cases per tumor type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--copy-data", action="store_true", help="Copy selected cases to segmentation_comparison folder")
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load config
    cfg = load_config(args.config)
    paths = cfg["paths"]
    
    # Setup output directories
    output_root = paths["output_root"]
    seg_comparison_dir = os.path.join(output_root, "segmentation_comparison")
    test_subset_dir = os.path.join(seg_comparison_dir, "test_subset")
    ground_truth_dir = os.path.join(seg_comparison_dir, "ground_truth_masks")
    
    os.makedirs(seg_comparison_dir, exist_ok=True)
    os.makedirs(ground_truth_dir, exist_ok=True)
    
    subset_csv_path = os.path.join(seg_comparison_dir, "segmentation_test_subset.csv")
    
    # Load master index
    print(f"Loading master index from {paths['master_csv']}")
    with open(paths["master_csv"], "r") as f:
        all_cases = list(csv.DictReader(f))
    
    print(f"Total cases in master index: {len(all_cases)}")
    
    # Filter criteria:
    # 1. Training split only (not validation/test)
    # 2. Has segmentation mask
    # 3. No missing modalities
    eligible_cases = [
        c for c in all_cases
        if c["split"] == "train"
        and c["has_seg"] == "yes"
        and c["has_missing"] == "no"
    ]
    
    print(f"Eligible cases (train, has_seg, no_missing): {len(eligible_cases)}")
    
    # Separate by tumor type
    gli_cases = [c for c in eligible_cases if c["dataset"] == "GLI"]
    men_cases = [c for c in eligible_cases if c["dataset"] == "MEN"]
    
    print(f"  GLI: {len(gli_cases)} cases")
    print(f"  MEN: {len(men_cases)} cases")
    
    # Check if we have enough cases
    if len(gli_cases) < args.n_per_class:
        print(f"Warning: Only {len(gli_cases)} GLI cases available, requested {args.n_per_class}")
        args.n_per_class = len(gli_cases)
    
    if len(men_cases) < args.n_per_class:
        print(f"Warning: Only {len(men_cases)} MEN cases available, requested {args.n_per_class}")
        args.n_per_class = len(men_cases)
    
    # Randomly sample
    selected_gli = random.sample(gli_cases, args.n_per_class)
    selected_men = random.sample(men_cases, args.n_per_class)
    
    selected_cases = selected_gli + selected_men
    
    print(f"\nSelected {len(selected_cases)} cases:")
    print(f"  GLI: {len(selected_gli)}")
    print(f"  MEN: {len(selected_men)}")
    
    # Save metadata CSV
    with open(subset_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=selected_cases[0].keys())
        writer.writeheader()
        writer.writerows(selected_cases)
    
    print(f"\nSaved subset metadata to: {subset_csv_path}")
    
    # Copy data if requested
    if args.copy_data:
        print("\nCopying selected cases to segmentation_comparison folder...")
        
        # Create subdirectories
        gli_dest = os.path.join(test_subset_dir, "GLI")
        men_dest = os.path.join(test_subset_dir, "MEN")
        os.makedirs(gli_dest, exist_ok=True)
        os.makedirs(men_dest, exist_ok=True)
        
        # Copy GLI cases
        print("\nCopying GLI cases...")
        for case in selected_gli:
            copy_case_data(
                case["case_id"], 
                "GLI",
                paths["gli_training_dir"],
                gli_dest,
                ground_truth_dir
            )
        
        # Copy MEN cases
        print("\nCopying MEN cases...")
        for case in selected_men:
            copy_case_data(
                case["case_id"],
                "MEN", 
                paths["men_training_dir"],
                men_dest,
                ground_truth_dir
            )
        
        print(f"\nData copied to: {seg_comparison_dir}")
        print(f"  Test subset:     {test_subset_dir}")
        print(f"  Ground truth:    {ground_truth_dir}")
    
    # Print statistics
    gli_sites = set(c["site"] for c in selected_gli if c["site"])
    men_has_site = any(c["site"] for c in selected_men)
    
    print(f"\nSubset statistics:")
    print(f"  GLI sites represented: {len(gli_sites) if gli_sites else 'N/A'}")
    print(f"  MEN site info: {'Available' if men_has_site else 'Not available'}")
    
    # Print sample case IDs
    print(f"\nSample case IDs:")
    print("GLI:")
    for c in selected_gli[:5]:
        print(f"  {c['case_id']}")
    if len(selected_gli) > 5:
        print(f"  ... ({len(selected_gli)} total)")
    
    print("\nMEN:")
    for c in selected_men[:5]:
        print(f"  {c['case_id']}")
    if len(selected_men) > 5:
        print(f"  ... ({len(selected_men)} total)")
    
    print("\n" + "="*60)
    print("Selection complete!")
    print(f"Output directory: {seg_comparison_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
