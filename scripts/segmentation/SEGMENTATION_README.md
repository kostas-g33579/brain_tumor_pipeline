# Brain Tumor Segmentation Comparison Study

## Overview

This study evaluates two state-of-the-art automated segmentation methods (TotalSegmentator and nnU-Net) against expert ground truth annotations for brain tumor segmentation. The goal was to determine whether automated methods could replace manual segmentations in the brain tumor classification pipeline.

## Methods Evaluated

### 1. TotalSegmentator
- **Type:** General-purpose whole-body segmentation
- **Input:** Single modality (T1-native)
- **Training data:** Healthy anatomy from CT/MRI scans
- **Output:** 117 organ masks (brain extracted)

### 2. nnU-Net V2
- **Type:** Brain tumor-specific segmentation
- **Input:** 4 modalities (T1, T1ce, T2, FLAIR)
- **Training data:** BraTS 2021 (gliomas only)
- **Pretrained model:** Dataset002_BRATS19 from Zenodo (DOI: 10.5281/zenodo.11582627)
- **Configuration:** 3d_fullres with 5-fold ensemble

### 3. Ground Truth
- Expert manual annotations from BraTS 2023 Challenge
- Multi-class labels: background, edema, non-enhancing tumor, enhancing tumor

## Test Dataset

- **Total cases:** 50 (25 glioma + 25 meningioma)
- **Source:** BraTS 2023 training set
- **Selection:** Random sampling with seed=42
- **Modalities:** T1-native, T1-contrast, T2-weighted, T2-FLAIR
- **Resolution:** Isotropic 1mm³, 240×240×155 voxels

## Evaluation Metrics

Following official BraTS Challenge methodology:

1. **Dice Similarity Coefficient (DSC)**
   - Formula: DSC = 2×|A∩B| / (|A| + |B|)
   - Range: 0 to 1 (higher is better)
   - Thresholds: >0.90 excellent, 0.70-0.90 good, <0.70 poor

2. **95th Percentile Hausdorff Distance (HD95)**
   - Measures boundary accuracy
   - Unit: millimeters
   - Lower is better

### Tumor Regions

- **WT (Whole Tumor):** All tumor tissue (labels 1, 2, 4)
- **TC (Tumor Core):** Non-enhancing + enhancing tumor (labels 1, 4)
- **ET (Enhancing Tumor):** Actively enhancing regions (label 4)

## Results

Results will be reported after evaluation on the full dataset.

### Evaluation Framework

The evaluation computes metrics for three tumor regions:
- **WT (Whole Tumor):** All tumor tissue
- **TC (Tumor Core):** Non-enhancing + enhancing tumor
- **ET (Enhancing Tumor):** Actively enhancing regions

Metrics are computed separately for:
- Glioma (GLI) cases
- Meningioma (MEN) cases
- Overall performance

## Preliminary Findings (50-case test subset)

1. TotalSegmentator is unsuitable for brain tumor segmentation (designed for healthy anatomy)
2. nnU-Net pretrained on BraTS 2021 shows limited generalization
3. Evaluation framework validated using official BraTS methodology
4. Full dataset evaluation pending

## Recommendation

**Use BraTS ground truth segmentations for the classification pipeline.**

Reasons:
- Gold standard quality (expert radiologists)
- Multi-class labels with detailed subregion information
- Already available in the dataset
- Validated in hundreds of published papers
- Consistent annotation protocol across all cases

## Implementation

### Environment Setup

```bash
# Create conda environment
conda create -n btc python=3.10
conda activate btc

# Install required packages
pip install totalsegmentator
pip install nnunetv2
pip install medpy nibabel pandas numpy matplotlib seaborn
```

### TotalSegmentator Execution

```bash
# Batch processing
python scripts/segmentation/02_run_totalsegmentator.py
```

### nnU-Net V2 Execution

```bash
# Set environment variables (Windows)
setx nnUNet_raw "D:\thesis\nnUNet_raw"
setx nnUNet_preprocessed "D:\thesis\nnUNet_preprocessed"
setx nnUNet_results "D:\thesis\nnUNet_results"

# Download pretrained model from Zenodo: 10.5281/zenodo.11582627
# Install model
nnUNetv2_install_pretrained_model_from_zip Dataset002_BRATS19.zip

# Prepare test data
python scripts/segmentation/03_prepare_nnunet_data.py

# Run inference
nnUNetv2_predict -i D:\thesis\nnUNet_input \
    -o D:\thesis\nnunet_output \
    -d Dataset002_BRATS19 -c 3d_fullres \
    -f 0 1 2 3 4 -device cuda
```

### Evaluation

```bash
# Compute metrics
python scripts/segmentation/04_evaluate_segmentation.py

# Generate figures
python create_segmentation_plots.py
```

## Output Files

- `segmentation_test_subset.csv` - List of 50 test cases with metadata
- `nnunet_evaluation_results.csv` - Per-case Dice and HD95 scores
- `nnunet_evaluation_summary.csv` - Summary statistics by dataset and region
- `segmentation_dice_boxplots.png/pdf` - Dice score distributions
- `segmentation_hd95_boxplots.png/pdf` - HD95 distributions
- `segmentation_summary_bars.png/pdf` - Bar chart with error bars
- `segmentation_heatmap.png/pdf` - Performance heatmap
- `segmentation_summary_table.csv/tex` - LaTeX-ready summary table

## References

1. Wasserthal J, et al. (2023). TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. Radiology: Artificial Intelligence, 5(5).

2. Isensee F, et al. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 18(2), 203-211.

3. Taha AA, Hanbury A. (2015). Metrics for evaluating 3D medical image segmentation. BMC Medical Imaging.

4. BraTS 2023 Challenge: https://www.synapse.org/brats2023

5. BraTS 2023 Official Metrics: https://github.com/rachitsaluja/BraTS-2023-Metrics

## Hardware Requirements

- GPU: NVIDIA RTX 3090 or equivalent (CUDA required for nnU-Net)
- RAM: 32 GB recommended
- Storage: ~15 GB for pretrained models and outputs

## License

This project is part of an MSc thesis at Aristotle University of Thessaloniki (AUTH).

## Author

Kostas Plomaritis  
MSc Biomedical Engineering  
Aristotle University of Thessaloniki  
February 2026
