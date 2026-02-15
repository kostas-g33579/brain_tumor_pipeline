"""
Prepare Test Data for nnU-Net V2 Inference
===========================================
Converts BraTS test subset to nnU-Net V2 format with correct modality ordering.

nnU-Net expects:
  0: t1    → BraTS-GLI-XXXXX-XXX_0000.nii.gz
  1: t1ce  → BraTS-GLI-XXXXX-XXX_0001.nii.gz
  2: t2    → BraTS-GLI-XXXXX-XXX_0002.nii.gz
  3: flair → BraTS-GLI-XXXXX-XXX_0003.nii.gz

BraTS naming:
  *-t1n.nii.gz  → modality 0 (t1)
  *-t1c.nii.gz  → modality 1 (t1ce)
  *-t2w.nii.gz  → modality 2 (t2)
  *-t2f.nii.gz  → modality 3 (flair)
"""

import csv
import shutil
from pathlib import Path


def prepare_nnunet_data():
    # Paths
    base_dir = Path("D:/thesis/thesis_outputs")
    test_subset_dir = base_dir / "test_subset"
    subset_csv = base_dir / "segmentation_test_subset.csv"
    
    # Output directory for nnU-Net formatted data
    nnunet_input_dir = Path("D:/thesis/nnUNet_input")
    nnunet_input_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("Preparing Test Data for nnU-Net V2")
    print("="*60)
    
    # Load test cases
    with open(subset_csv, "r") as f:
        cases = list(csv.DictReader(f))
    
    print(f"\nTotal cases to prepare: {len(cases)}")
    print(f"Output directory: {nnunet_input_dir}\n")
    
    # Modality mapping: BraTS suffix → nnU-Net modality index
    modality_map = {
        "t1n": "0000",  # t1
        "t1c": "0001",  # t1ce
        "t2w": "0002",  # t2
        "t2f": "0003",  # flair
    }
    
    prepared = 0
    skipped = 0
    
    for i, case in enumerate(cases):
        case_id = case["case_id"]
        dataset = case["dataset"]
        
        case_dir = test_subset_dir / dataset / case_id
        
        # Check if all 4 modalities exist
        all_exist = True
        for suffix in ["t1n", "t1c", "t2w", "t2f"]:
            src_file = case_dir / f"{case_id}-{suffix}.nii.gz"
            if not src_file.exists():
                print(f"[{i+1}/{len(cases)}] SKIP: {case_id} - Missing {suffix}")
                all_exist = False
                skipped += 1
                break
        
        if not all_exist:
            continue
        
        # Copy and rename all 4 modalities
        print(f"[{i+1}/{len(cases)}] Preparing: {case_id}")
        
        for brats_suffix, nnunet_suffix in modality_map.items():
            src_file = case_dir / f"{case_id}-{brats_suffix}.nii.gz"
            dst_file = nnunet_input_dir / f"{case_id}_{nnunet_suffix}.nii.gz"
            
            shutil.copy2(src_file, dst_file)
        
        prepared += 1
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Prepared:  {prepared} cases ({prepared * 4} files)")
    print(f"Skipped:   {skipped} cases")
    print(f"Output:    {nnunet_input_dir}")
    print("="*60)
    
    print("\n✓ Data preparation complete!")
    print(f"\nNext step: Run nnU-Net inference with:")
    print(f"  nnUNetv2_predict -i {nnunet_input_dir} -o D:/thesis/nnunet_output \\")
    print(f"    -d Dataset002_BRATS19 -c 3d_fullres -f 0 1 2 3 4")


if __name__ == "__main__":
    prepare_nnunet_data()
