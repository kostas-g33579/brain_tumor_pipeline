"""
TotalSegmentator Batch Processing
==================================
Runs TotalSegmentator on all 50 test cases (25 GLI + 25 MEN).

Output structure:
  totalsegmentator_output/
    ├── BraTS-GLI-00284-000_totalseg.nii.gz
    ├── BraTS-GLI-00054-000_totalseg.nii.gz
    └── ...

Usage:
    python scripts/segmentation/02_run_totalsegmentator.py
"""

import os
import subprocess
import csv
import time
from pathlib import Path
from datetime import datetime


def run_totalsegmentator(input_path, output_path, device="gpu"):
    """
    Run TotalSegmentator on a single case.
    
    Returns:
        (success, duration, error_msg)
    """
    start_time = time.time()
    
    try:
        # Run TotalSegmentator
        result = subprocess.run(
            [
                "TotalSegmentator",
                "-i", input_path,
                "-o", output_path,
                "--device", device,
                "-q"  # Quiet mode (less verbose)
            ],
            capture_output=True,
            text=True,
            check=True
        )
        
        duration = time.time() - start_time
        return True, duration, None
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        return False, duration, str(e)
    except Exception as e:
        duration = time.time() - start_time
        return False, duration, str(e)


def main():
    print("="*60)
    print("TotalSegmentator Batch Processing")
    print("="*60)
    
    # Paths
    base_dir = Path("D:/thesis/thesis_outputs")
    test_subset_dir = base_dir / "test_subset"
    output_base = base_dir / "totalsegmentator_output"
    output_base.mkdir(exist_ok=True)
    
    log_file = output_base / f"totalsegmentator_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Load test subset metadata
    subset_csv = base_dir / "segmentation_test_subset.csv"
    with open(subset_csv, "r") as f:
        cases = list(csv.DictReader(f))
    
    print(f"\nTotal cases to process: {len(cases)}")
    print(f"Output directory: {output_base}")
    print(f"Log file: {log_file}\n")
    
    # Process each case
    results = []
    total_start = time.time()
    
    for i, case in enumerate(cases):
        case_id = case["case_id"]
        dataset = case["dataset"]
        
        # Input path (use T1 native as input)
        input_file = test_subset_dir / dataset / case_id / f"{case_id}-t1n.nii.gz"
        
        if not input_file.exists():
            print(f"[{i+1}/{len(cases)}] SKIP: {case_id} - Input file not found")
            results.append({
                "case_id": case_id,
                "dataset": dataset,
                "status": "skipped",
                "duration": 0,
                "error": "Input file not found"
            })
            continue
        
        # Output path (temporary folder for this case)
        temp_output = output_base / f"{case_id}_temp"
        final_output = output_base / f"{case_id}_totalseg.nii.gz"
        
        print(f"[{i+1}/{len(cases)}] Processing: {case_id} ({dataset})...")
        
        # Run TotalSegmentator
        success, duration, error = run_totalsegmentator(
            str(input_file),
            str(temp_output),
            device="gpu"
        )
        
        if success:
            # TotalSegmentator creates individual organ masks
            # For brain tumor segmentation, we only need brain.nii.gz
            
            # Find the brain mask
            brain_mask = temp_output / "brain.nii.gz"
            if brain_mask.exists():
                # Move to final location
                brain_mask.rename(final_output)
                
                # Clean up temp folder
                import shutil
                shutil.rmtree(temp_output)
                
                print(f"  ✓ Success ({duration:.1f}s)")
                results.append({
                    "case_id": case_id,
                    "dataset": dataset,
                    "status": "success",
                    "duration": duration,
                    "error": None
                })
            else:
                print(f"  ✗ Failed: Brain mask not found")
                results.append({
                    "case_id": case_id,
                    "dataset": dataset,
                    "status": "failed",
                    "duration": duration,
                    "error": "Brain mask not found"
                })
        else:
            print(f"  ✗ Failed ({duration:.1f}s): {error}")
            results.append({
                "case_id": case_id,
                "dataset": dataset,
                "status": "failed",
                "duration": duration,
                "error": error
            })
    
    # Summary
    total_duration = time.time() - total_start
    n_success = sum(1 for r in results if r["status"] == "success")
    n_failed = sum(1 for r in results if r["status"] == "failed")
    n_skipped = sum(1 for r in results if r["status"] == "skipped")
    
    avg_duration = sum(r["duration"] for r in results if r["status"] == "success") / max(n_success, 1)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total cases:    {len(cases)}")
    print(f"  Success:      {n_success}")
    print(f"  Failed:       {n_failed}")
    print(f"  Skipped:      {n_skipped}")
    print(f"\nTotal time:     {total_duration/60:.1f} minutes")
    print(f"Avg per case:   {avg_duration:.1f} seconds")
    print(f"\nOutput folder:  {output_base}")
    print("="*60)
    
    # Save log
    with open(log_file, "w") as f:
        f.write("TotalSegmentator Batch Processing Log\n")
        f.write("="*60 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total cases: {len(cases)}\n")
        f.write(f"Success: {n_success}, Failed: {n_failed}, Skipped: {n_skipped}\n")
        f.write(f"Total time: {total_duration/60:.1f} minutes\n")
        f.write(f"Avg per case: {avg_duration:.1f} seconds\n")
        f.write("\n" + "="*60 + "\n\n")
        
        for r in results:
            status_symbol = "✓" if r["status"] == "success" else ("✗" if r["status"] == "failed" else "⊘")
            f.write(f"{status_symbol} {r['case_id']} ({r['dataset']}): {r['status']} ({r['duration']:.1f}s)\n")
            if r["error"]:
                f.write(f"    Error: {r['error']}\n")
    
    print(f"\nLog saved: {log_file}")


if __name__ == "__main__":
    main()
