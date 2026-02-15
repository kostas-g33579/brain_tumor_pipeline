#!/usr/bin/env Rscript
################################################################################
# Step 5 — ComBat Harmonization
################################################################################
# 
# Removes site/batch effects from radiomics features while preserving 
# biological variation (tumor type: glioma vs meningioma).
#
# Methodology follows the official ComBat guidelines:
# - Johnson et al. (2007) "Adjusting batch effects in microarray expression data"
# - Fortin et al. (2017) "Harmonization of multi-site diffusion tensor imaging data"
# 
# ComBat uses empirical Bayes to:
#   1. Estimate location and scale batch effects
#   2. Pool information across features
#   3. Adjust feature distributions while preserving biological signal
#
# Usage:
#   Rscript 05_combat_harmonization.R <input_csv> <output_csv> <batch_col> <bio_covariates>
#
# Example:
#   Rscript 05_combat_harmonization.R \
#     features_clean.csv \
#     features_harmonized.csv \
#     site \
#     label
#
################################################################################

library(sva)      # Contains ComBat
library(readr)    # Fast CSV reading
library(dplyr)    # Data manipulation

################################################################################
# Configuration
################################################################################

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3) {
  cat("Usage: Rscript 05_combat_harmonization.R <input_csv> <output_csv> <batch_col> [bio_covariates]\n")
  cat("Example: Rscript 05_combat_harmonization.R features_clean.csv features_harmonized.csv site label\n")
  cat("Example (no covariates): Rscript 05_combat_harmonization.R features_clean.csv features_harmonized.csv site\n")
  quit(status = 1)
}

input_csv <- args[1]
output_csv <- args[2]
batch_col <- args[3]
bio_covariates <- if (length(args) >= 4 && args[4] != "") strsplit(args[4], ",")[[1]] else c()  # Empty if not provided or empty string

cat("================================================================\n")
cat("ComBat Harmonization - Brain Tumor Classification\n")
cat("================================================================\n")
cat("Input:             ", input_csv, "\n")
cat("Output:            ", output_csv, "\n")
cat("Batch variable:    ", batch_col, "\n")
cat("Bio covariates:    ", paste(bio_covariates, collapse = ", "), "\n")
cat("================================================================\n\n")

################################################################################
# Load and prepare data
################################################################################

cat("Loading feature matrix...\n")
df <- read_csv(input_csv, show_col_types = FALSE)
cat("  Loaded:", nrow(df), "cases ×", ncol(df), "columns\n\n")

# Identify metadata columns (not features)
meta_cols <- c("case_id", "label", "label_name", "dataset", "split", "site", 
               "cohort", "batch", "institution")
meta_cols <- intersect(meta_cols, colnames(df))

# Feature columns are everything except metadata
feature_cols <- setdiff(colnames(df), meta_cols)
cat("  Meta columns:   ", length(meta_cols), "\n")
cat("  Feature columns:", length(feature_cols), "\n\n")

# Extract batch variable
if (!batch_col %in% colnames(df)) {
  stop("Error: Batch column '", batch_col, "' not found in data")
}
batch <- df[[batch_col]]

# Convert batch to factor and handle missing values
batch <- as.factor(batch)
if (any(is.na(batch))) {
  stop("Error: Batch variable contains NA values. All cases must have a batch assignment.")
}

cat("Batch distribution:\n")
print(table(batch))
cat("\n")

# Check minimum batch size (ComBat requires at least 2 samples per batch)
batch_sizes <- table(batch)
if (any(batch_sizes < 2)) {
  small_batches <- names(batch_sizes[batch_sizes < 2])
  stop("Error: The following batches have <2 samples: ", paste(small_batches, collapse = ", "))
}

################################################################################
# Prepare biological covariates matrix
################################################################################

# Build model matrix for biological variables we want to preserve
# This ensures ComBat doesn't remove biologically meaningful variation
if (length(bio_covariates) > 0) {
  cat("Building covariate matrix to preserve biological variation...\n")
  
  covariate_formula <- as.formula(paste("~", paste(bio_covariates, collapse = " + ")))
  mod <- model.matrix(covariate_formula, data = df)
  
  cat("  Covariates preserved:", paste(bio_covariates, collapse = ", "), "\n")
  cat("  Model matrix dimensions:", nrow(mod), "×", ncol(mod), "\n\n")
} else {
  # No biological covariates (parametric adjustment only)
  mod <- model.matrix(~ 1, data = df)
  cat("No biological covariates specified (NULL model)\n\n")
}

################################################################################
# Prepare feature matrix for ComBat
################################################################################

# Extract feature matrix (samples × features)
feature_matrix <- as.matrix(df[, feature_cols])

# ComBat expects features × samples, so transpose
feature_matrix_t <- t(feature_matrix)

cat("Feature matrix prepared:\n")
cat("  Dimensions (transposed):", nrow(feature_matrix_t), "features ×", ncol(feature_matrix_t), "samples\n")
cat("  Range: [", min(feature_matrix, na.rm = TRUE), ",", max(feature_matrix, na.rm = TRUE), "]\n\n")

# Check for missing values
n_missing <- sum(is.na(feature_matrix))
if (n_missing > 0) {
  cat("Warning:", n_missing, "missing values detected in feature matrix\n")
  cat("  ComBat will skip features with missing values\n\n")
}

################################################################################
# Run ComBat harmonization
################################################################################

cat("Running ComBat harmonization...\n")
cat("  Method: Empirical Bayes (parametric)\n")
cat("  Preserving biological variation: YES\n\n")

# Run ComBat with parametric empirical Bayes adjustments
# par.prior = TRUE: Use parametric empirical Bayes (recommended for >25 samples)
# mean.only = FALSE: Adjust both location (mean) and scale (variance)

harmonized_matrix_t <- ComBat(
  dat = feature_matrix_t,
  batch = batch,
  mod = mod,
  par.prior = TRUE,
  mean.only = FALSE,
  ref.batch = NULL  # No reference batch (adjust all batches equally)
)

cat("\nComBat harmonization complete.\n\n")

################################################################################
# Transpose back and reconstruct dataframe
################################################################################

# Transpose harmonized features back to samples × features
harmonized_matrix <- t(harmonized_matrix_t)

# Verify dimensions match
if (nrow(harmonized_matrix) != nrow(df) || ncol(harmonized_matrix) != length(feature_cols)) {
  stop("Error: Harmonized matrix dimensions don't match original")
}

# Reconstruct full dataframe with metadata + harmonized features
df_harmonized <- df[, meta_cols]  # Keep metadata unchanged
df_harmonized <- cbind(df_harmonized, as.data.frame(harmonized_matrix))

# Verify column order matches original
if (!all(colnames(df_harmonized)[-seq_along(meta_cols)] == feature_cols)) {
  stop("Error: Column order mismatch after harmonization")
}

cat("Harmonized feature matrix reconstructed:\n")
cat("  Dimensions:", nrow(df_harmonized), "cases ×", ncol(df_harmonized), "columns\n")
cat("  Meta columns:", length(meta_cols), "\n")
cat("  Feature columns:", length(feature_cols), "\n\n")

################################################################################
# Quality checks
################################################################################

cat("Quality checks:\n")

# Check for NAs introduced by ComBat
n_missing_after <- sum(is.na(df_harmonized[, feature_cols]))
if (n_missing_after > n_missing) {
  cat("  Warning: ComBat introduced", n_missing_after - n_missing, "additional missing values\n")
}

# Check feature ranges
feature_ranges_before <- apply(feature_matrix, 2, function(x) diff(range(x, na.rm = TRUE)))
feature_ranges_after <- apply(harmonized_matrix, 2, function(x) diff(range(x, na.rm = TRUE)))

n_zero_variance <- sum(feature_ranges_after < 1e-10, na.rm = TRUE)
if (n_zero_variance > 0) {
  cat("  Warning:", n_zero_variance, "features have near-zero variance after harmonization\n")
}

cat("  Range before: [", min(feature_matrix, na.rm = TRUE), ",", max(feature_matrix, na.rm = TRUE), "]\n")
cat("  Range after:  [", min(harmonized_matrix, na.rm = TRUE), ",", max(harmonized_matrix, na.rm = TRUE), "]\n")
cat("\n")

################################################################################
# Save harmonized features
################################################################################

cat("Saving harmonized feature matrix...\n")
write_csv(df_harmonized, output_csv)
cat("  Saved:", output_csv, "\n\n")

################################################################################
# Summary report
################################################################################

cat("================================================================\n")
cat("ComBat Harmonization Summary\n")
cat("================================================================\n")
cat("Input cases:        ", nrow(df), "\n")
cat("Output cases:       ", nrow(df_harmonized), "\n")
cat("Features harmonized:", length(feature_cols), "\n")
cat("Batches adjusted:   ", nlevels(batch), "\n")
cat("Biological vars preserved:", paste(bio_covariates, collapse = ", "), "\n")
cat("Output file:        ", output_csv, "\n")
cat("================================================================\n")
cat("\nComBat harmonization complete.\n")
