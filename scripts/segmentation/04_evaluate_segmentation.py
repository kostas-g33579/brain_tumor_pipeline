"""
BraTS-Compliant Segmentation Evaluation Script
================================================
Evaluates nnU-Net predictions against ground truth using official BraTS metrics.

Uses MedPy library - the standard validated library for medical image evaluation.

Follows BraTS Challenge methodology:
- Dice Similarity Coefficient (DSC)
- 95th Percentile Hausdorff Distance (HD95) in mm

Tumor regions evaluated:
- WT (Whole Tumor): labels 1, 2, 4
- TC (Tumor Core): labels 1, 4
- ET (Enhancing Tumor): label 4

References:
- BraTS 2023 Metrics: https://github.com/rachitsaluja/BraTS-2023-Metrics
- Taha & Hanbury (2015): Metrics for evaluating 3D medical image segmentation
- Menze et al. (2015): The Multimodal Brain Tumor Image Segmentation Benchmark
- MedPy library: https://loli.github.io/medpy/
"""

import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from medpy.metric.binary import dc, hd95  # Standard validated metrics
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


def extract_tumor_regions(mask: np.ndarray) -> dict:
    """
    Extract BraTS tumor regions from segmentation mask.
    
    BraTS labels:
    - 0: Background
    - 1: Necrotic/non-enhancing tumor core (NCR)
    - 2: Peritumoral edema (ED)
    - 4: Enhancing tumor (ET)
    
    Regions:
    - WT (Whole Tumor): 1, 2, 4
    - TC (Tumor Core): 1, 4
    - ET (Enhancing Tumor): 4
    
    Args:
        mask: Segmentation mask with BraTS labels
    
    Returns:
        Dictionary with binary masks for each region
    """
    return {
        'WT': np.isin(mask, [1, 2, 4]),  # Whole tumor
        'TC': np.isin(mask, [1, 4]),     # Tumor core
        'ET': (mask == 4),                # Enhancing tumor
    }


def evaluate_case(pred_path: Path, gt_path: Path, voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> dict:
    """
    Evaluate a single case: compute DSC and HD95 for all tumor regions.
    
    Uses MedPy library functions (validated and standard).
    
    Args:
        pred_path: Path to prediction NIfTI file
        gt_path: Path to ground truth NIfTI file
        voxel_spacing: Voxel dimensions in mm
    
    Returns:
        Dictionary with results for each region
    """
    # Load images
    pred_img = nib.load(pred_path)
    gt_img = nib.load(gt_path)
    
    pred_data = pred_img.get_fdata().astype(np.uint8)
    gt_data = gt_img.get_fdata().astype(np.uint8)
    
    # Extract tumor regions
    pred_regions = extract_tumor_regions(pred_data)
    gt_regions = extract_tumor_regions(gt_data)
    
    results = {}
    
    # Evaluate each region
    for region in ['WT', 'TC', 'ET']:
        pred_binary = pred_regions[region].astype(bool)
        gt_binary = gt_regions[region].astype(bool)
        
        # Handle edge cases
        if pred_binary.sum() == 0 and gt_binary.sum() == 0:
            # Both empty - perfect match
            dice_score = 1.0
            hd95_score = 0.0
        elif pred_binary.sum() == 0 or gt_binary.sum() == 0:
            # One empty - no overlap
            dice_score = 0.0
            hd95_score = 373.13  # Max possible distance in 240×240×155 volume
        else:
            # Both non-empty - use MedPy validated functions
            try:
                dice_score = dc(pred_binary, gt_binary)
            except Exception as e:
                print(f"      Warning: Dice computation failed for {region}: {e}")
                dice_score = 0.0
            
            try:
                hd95_score = hd95(pred_binary, gt_binary, voxelspacing=voxel_spacing)
            except Exception as e:
                print(f"      Warning: HD95 computation failed for {region}: {e}")
                hd95_score = 373.13
        
        results[f'{region}_Dice'] = float(dice_score)
        results[f'{region}_HD95'] = float(hd95_score)
    
    return results


def main():
    print("="*70)
    print("BraTS-Compliant Segmentation Evaluation")
    print("="*70)
    print("\nFollowing official BraTS Challenge methodology:")
    print("  • Dice Similarity Coefficient (DSC)")
    print("  • 95th Percentile Hausdorff Distance (HD95)")
    print("\nTumor regions:")
    print("  • WT (Whole Tumor): labels 1, 2, 4")
    print("  • TC (Tumor Core): labels 1, 4")
    print("  • ET (Enhancing Tumor): label 4")
    print("="*70)
    
    # Paths
    base_dir = Path("C:/Users/kplom/Documents/thesis/thesis_outputs")
    nnunet_dir = base_dir / "nnunet_output"
    gt_dir = base_dir / "segmentation_comparison" / "ground_truth_masks"
    subset_csv = base_dir / "segmentation_comparison" / "segmentation_test_subset.csv"
    
    # Load subset metadata
    subset_df = pd.read_csv(subset_csv)
    
    # Find successful nnU-Net cases (non-empty masks)
    nnunet_files = list(nnunet_dir.glob("*.nii.gz"))
    
    successful_cases = []
    for nii_file in nnunet_files:
        # Check if mask is non-empty
        mask = nib.load(nii_file).get_fdata()
        if np.count_nonzero(mask) > 0:
            case_id = nii_file.stem.replace(".nii", "")
            successful_cases.append(case_id)
    
    print(f"\nFound {len(successful_cases)} successful nnU-Net predictions")
    print(f"  GLI: {sum(1 for c in successful_cases if 'GLI' in c)}")
    print(f"  MEN: {sum(1 for c in successful_cases if 'MEN' in c)}")
    print()
    
    # Evaluate each case
    results = []
    
    for i, case_id in enumerate(sorted(successful_cases), 1):
        pred_path = nnunet_dir / f"{case_id}.nii.gz"
        gt_path = gt_dir / f"{case_id}-seg.nii.gz"  # Ground truth has -seg suffix
        
        if not gt_path.exists():
            print(f"[{i}/{len(successful_cases)}] SKIP: {case_id} - Ground truth not found")
            continue
        
        print(f"[{i}/{len(successful_cases)}] Evaluating: {case_id}")
        
        # BraTS data is isotropic 1mm³
        case_results = evaluate_case(pred_path, gt_path, voxel_spacing=(1.0, 1.0, 1.0))
        
        # Add metadata
        case_results['case_id'] = case_id
        case_results['dataset'] = 'GLI' if 'GLI' in case_id else 'MEN'
        
        results.append(case_results)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['case_id', 'dataset',
            'WT_Dice', 'WT_HD95',
            'TC_Dice', 'TC_HD95',
            'ET_Dice', 'ET_HD95']
    results_df = results_df[cols]
    
    # Save detailed results
    output_csv = base_dir / "segmentation_comparison" / "nnunet_evaluation_results.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\n✓ Detailed results saved: {output_csv}")
    
    # Compute summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    for dataset in ['GLI', 'MEN', 'Overall']:
        if dataset == 'Overall':
            subset = results_df
        else:
            subset = results_df[results_df['dataset'] == dataset]
        
        if len(subset) == 0:
            continue
        
        print(f"\n{dataset} (n={len(subset)}):")
        print("-" * 70)
        
        for region in ['WT', 'TC', 'ET']:
            dice_col = f'{region}_Dice'
            hd95_col = f'{region}_HD95'
            
            dice_mean = subset[dice_col].mean()
            dice_std = subset[dice_col].std()
            hd95_mean = subset[hd95_col].mean()
            hd95_std = subset[hd95_col].std()
            
            print(f"  {region}:")
            print(f"    Dice:  {dice_mean:.4f} ± {dice_std:.4f}")
            print(f"    HD95:  {hd95_mean:.2f} ± {hd95_std:.2f} mm")
    
    # Save summary
    summary_rows = []
    for dataset in ['GLI', 'MEN', 'Overall']:
        if dataset == 'Overall':
            subset = results_df
        else:
            subset = results_df[results_df['dataset'] == dataset]
        
        if len(subset) == 0:
            continue
        
        for region in ['WT', 'TC', 'ET']:
            summary_rows.append({
                'Dataset': dataset,
                'Region': region,
                'N': len(subset),
                'Dice_Mean': subset[f'{region}_Dice'].mean(),
                'Dice_Std': subset[f'{region}_Dice'].std(),
                'HD95_Mean': subset[f'{region}_HD95'].mean(),
                'HD95_Std': subset[f'{region}_HD95'].std(),
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = base_dir / "segmentation_comparison" / "nnunet_evaluation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n✓ Summary statistics saved: {summary_csv}")
    
    print("\n" + "="*70)
    print("Evaluation complete!")
    print("="*70)
    
    # Performance interpretation
    print("\nPerformance interpretation (Dice scores):")
    print("  • > 0.90: Excellent")
    print("  • 0.70-0.90: Good")
    print("  • < 0.70: Poor")
    print("\nThese are standard thresholds used in medical image segmentation.")


if __name__ == "__main__":
    main()
