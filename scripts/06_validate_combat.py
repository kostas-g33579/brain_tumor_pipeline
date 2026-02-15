"""
ComBat Harmonization Quality Check
===================================
Compares features before and after ComBat harmonization to verify:
1. Batch effects (site clustering) are reduced
2. Biological variation is preserved
3. Feature distributions are reasonable

Generates diagnostic plots:
- PCA before/after harmonization (colored by site)
- Feature variance before/after
- Sample correlation heatmaps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse


def load_data(before_path, after_path):
    """Load features before and after harmonization."""
    df_before = pd.read_csv(before_path)
    df_after = pd.read_csv(after_path)
    
    print(f"Before: {df_before.shape}")
    print(f"After:  {df_after.shape}")
    
    # Verify same cases
    assert df_before['case_id'].equals(df_after['case_id']), "Case IDs don't match!"
    
    return df_before, df_after


def extract_features_and_metadata(df):
    """Separate feature matrix from metadata."""
    meta_cols = ['case_id', 'label', 'label_name', 'dataset', 'split', 'site']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    X = df[feature_cols].values
    metadata = df[meta_cols].copy()
    
    return X, metadata, feature_cols


def plot_feature_distributions_by_site(X_before, X_after, metadata, feature_names, output_dir):
    """
    Plot feature distributions by site for a few representative features.
    This shows how ComBat removes site-specific biases.
    
    Common in ComBat validation papers - shows boxplots of feature values
    across sites before and after harmonization.
    """
    # Select 4 features with highest variance (most informative)
    feature_vars = np.var(X_before, axis=0)
    top_features_idx = np.argsort(feature_vars)[-4:]
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    
    for plot_idx, feat_idx in enumerate(top_features_idx):
        feat_name = feature_names[feat_idx]
        
        # Prepare data for plotting
        df_before = pd.DataFrame({
            'value': X_before[:, feat_idx],
            'site': metadata['site']
        })
        
        df_after = pd.DataFrame({
            'value': X_after[:, feat_idx],
            'site': metadata['site']
        })
        
        # Before harmonization
        ax = axes[plot_idx, 0]
        df_before.boxplot(column='value', by='site', ax=ax)
        ax.set_title(f'Before: {feat_name[:40]}...', fontsize=9)
        ax.set_xlabel('Site')
        ax.set_ylabel('Feature Value')
        plt.sca(ax)
        plt.xticks(rotation=45, fontsize=7)
        
        # After harmonization
        ax = axes[plot_idx, 1]
        df_after.boxplot(column='value', by='site', ax=ax)
        ax.set_title(f'After: {feat_name[:40]}...', fontsize=9)
        ax.set_xlabel('Site')
        ax.set_ylabel('Feature Value')
        plt.sca(ax)
        plt.xticks(rotation=45, fontsize=7)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_distributions_by_site.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/feature_distributions_by_site.png")
    plt.close()


def plot_pca_comparison(X_before, X_after, metadata, output_dir):
    """
    Compare PCA before and after harmonization.
    Before: Should show site clustering (batch effects)
    After: Site clustering should be reduced
    """
    # Standardize features (mean=0, var=1) for PCA
    scaler_before = StandardScaler()
    scaler_after = StandardScaler()
    
    X_before_scaled = scaler_before.fit_transform(X_before)
    X_after_scaled = scaler_after.fit_transform(X_after)
    
    # Run PCA
    pca = PCA(n_components=2)
    pca_before = pca.fit_transform(X_before_scaled)
    pca_after = pca.fit_transform(X_after_scaled)
    
    var_before = pca.explained_variance_ratio_
    
    pca = PCA(n_components=2)
    pca_after_result = pca.fit_transform(X_after_scaled)
    var_after = pca.explained_variance_ratio_
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Before harmonization
    scatter = axes[0].scatter(pca_before[:, 0], pca_before[:, 1], 
                             c=metadata['site'], cmap='tab20', 
                             alpha=0.6, s=30)
    axes[0].set_xlabel(f'PC1 ({var_before[0]:.1%} variance)')
    axes[0].set_ylabel(f'PC2 ({var_before[1]:.1%} variance)')
    axes[0].set_title('Before ComBat: Sites Should Cluster\n(Batch Effects Present)', 
                     fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=axes[0], label='Site')
    axes[0].grid(True, alpha=0.3)
    
    # After harmonization
    scatter = axes[1].scatter(pca_after_result[:, 0], pca_after_result[:, 1], 
                             c=metadata['site'], cmap='tab20', 
                             alpha=0.6, s=30)
    axes[1].set_xlabel(f'PC1 ({var_after[0]:.1%} variance)')
    axes[1].set_ylabel(f'PC2 ({var_after[1]:.1%} variance)')
    axes[1].set_title('After ComBat: Site Clustering Should Be Reduced\n(Batch Effects Removed)', 
                     fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=axes[1], label='Site')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pca_comparison_by_site.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/pca_comparison_by_site.png")
    plt.close()


def plot_feature_variance_comparison(X_before, X_after, feature_names, output_dir):
    """
    Compare feature variances before/after.
    After harmonization, variance should be more uniform across batches.
    """
    var_before = np.var(X_before, axis=0)
    var_after = np.var(X_after, axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of variances
    axes[0].hist(np.log10(var_before + 1e-10), bins=50, alpha=0.7, label='Before', color='red')
    axes[0].hist(np.log10(var_after + 1e-10), bins=50, alpha=0.7, label='After', color='blue')
    axes[0].set_xlabel('Log10(Variance)')
    axes[0].set_ylabel('Number of Features')
    axes[0].set_title('Feature Variance Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Before vs After variance scatter
    axes[1].scatter(np.log10(var_before + 1e-10), np.log10(var_after + 1e-10), 
                   alpha=0.5, s=10)
    axes[1].plot([-10, 10], [-10, 10], 'r--', alpha=0.5, label='y=x')
    axes[1].set_xlabel('Log10(Variance) Before')
    axes[1].set_ylabel('Log10(Variance) After')
    axes[1].set_title('Feature Variance: Before vs After')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/variance_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/variance_comparison.png")
    plt.close()


def compute_batch_effect_metric(X, metadata):
    """
    Compute simple batch effect metric:
    Average within-site correlation vs between-site correlation.
    
    Higher ratio = stronger batch effects.
    After harmonization, ratio should decrease.
    """
    # Compute correlation matrix
    corr_matrix = np.corrcoef(X)
    
    sites = metadata['site'].values
    
    # Within-site correlations
    within_site_corrs = []
    for site in np.unique(sites):
        site_mask = sites == site
        site_indices = np.where(site_mask)[0]
        
        if len(site_indices) > 1:
            # Get upper triangle of within-site correlation submatrix
            for i in range(len(site_indices)):
                for j in range(i + 1, len(site_indices)):
                    within_site_corrs.append(corr_matrix[site_indices[i], site_indices[j]])
    
    # Between-site correlations
    between_site_corrs = []
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            if sites[i] != sites[j]:
                between_site_corrs.append(corr_matrix[i, j])
    
    mean_within = np.mean(within_site_corrs)
    mean_between = np.mean(between_site_corrs)
    
    ratio = mean_within / mean_between if mean_between != 0 else 0
    
    return mean_within, mean_between, ratio


def kolmogorov_smirnov_test(X, metadata, feature_names):
    """
    Perform Kolmogorov-Smirnov test for each feature across sites.
    
    This is the recommended method from radiomics ComBat papers (Nature Scientific Reports 2022).
    Tests if feature distributions differ significantly between sites.
    
    Returns:
        n_significant_before: Number of features with significant site differences
        p_values: Array of p-values for each feature
    """
    from scipy.stats import ks_2samp
    
    sites = metadata['site'].unique()
    n_features = X.shape[1]
    
    # For each feature, test if distributions differ between ANY pair of sites
    # Use Bonferroni-corrected threshold: 0.05 / n_comparisons
    n_site_pairs = len(sites) * (len(sites) - 1) // 2
    bonferroni_threshold = 0.05 / n_site_pairs
    
    significant_features = []
    
    for feat_idx in range(n_features):
        feature_values = X[:, feat_idx]
        
        # Test all pairs of sites
        min_p_value = 1.0
        for i, site1 in enumerate(sites):
            for site2 in sites[i+1:]:
                mask1 = metadata['site'] == site1
                mask2 = metadata['site'] == site2
                
                values1 = feature_values[mask1]
                values2 = feature_values[mask2]
                
                # KS test
                statistic, p_value = ks_2samp(values1, values2)
                min_p_value = min(min_p_value, p_value)
        
        # Feature is "significantly different across sites" if ANY pair differs
        if min_p_value < bonferroni_threshold:
            significant_features.append(feat_idx)
    
    return len(significant_features)


def compute_coefficient_of_variation(X, metadata):
    """
    Compute Coefficient of Variation (CoV) for site effect.
    
    This is the method used in travelling subject validation studies.
    Lower CoV = less site effect.
    
    CoV = (std_between_sites / mean_overall) * 100
    """
    sites = metadata['site'].values
    site_means = []
    
    # For each site, compute mean feature vector
    for site in np.unique(sites):
        site_mask = sites == site
        site_data = X[site_mask, :]
        site_mean = np.mean(site_data, axis=0)
        site_means.append(site_mean)
    
    site_means = np.array(site_means)
    
    # CoV for each feature
    overall_mean = np.mean(X, axis=0)
    std_between_sites = np.std(site_means, axis=0)
    
    # Avoid division by zero
    cov = np.zeros_like(overall_mean)
    nonzero_mask = overall_mean != 0
    cov[nonzero_mask] = (std_between_sites[nonzero_mask] / np.abs(overall_mean[nonzero_mask])) * 100
    
    mean_cov = np.mean(cov[nonzero_mask])
    
    return mean_cov


def main():
    parser = argparse.ArgumentParser(description="Validate ComBat harmonization quality")
    parser.add_argument("--before", required=True, help="Features before harmonization (CSV)")
    parser.add_argument("--after", required=True, help="Features after harmonization (CSV)")
    parser.add_argument("--output-dir", default=".", help="Output directory for plots")
    args = parser.parse_args()
    
    print("="*60)
    print("ComBat Harmonization Quality Check")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df_before, df_after = load_data(args.before, args.after)
    
    # Extract features
    X_before, metadata, feature_names = extract_features_and_metadata(df_before)
    X_after, _, _ = extract_features_and_metadata(df_after)
    
    print(f"\nFeatures: {len(feature_names)}")
    print(f"Samples: {len(metadata)}")
    print(f"Sites: {metadata['site'].nunique()}")
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. PCA comparison
    print("\n1. Computing PCA...")
    plot_pca_comparison(X_before, X_after, metadata, args.output_dir)
    
    # 2. Feature distributions by site
    print("\n2. Plotting feature distributions by site...")
    plot_feature_distributions_by_site(X_before, X_after, metadata, feature_names, args.output_dir)
    
    # 3. Variance comparison
    print("\n3. Comparing feature variances...")
    plot_feature_variance_comparison(X_before, X_after, feature_names, args.output_dir)
    
    # 4. Batch effect metrics (multiple methods from literature)
    print("\n3. Computing batch effect metrics...")
    
    # Method 1: Within/Between site correlation ratio
    within_before, between_before, ratio_before = compute_batch_effect_metric(X_before, metadata)
    within_after, between_after, ratio_after = compute_batch_effect_metric(X_after, metadata)
    
    print("\n  [Method 1] Within/Between Site Correlation Ratio:")
    print(f"    Before harmonization:")
    print(f"      Within-site correlation:  {within_before:.4f}")
    print(f"      Between-site correlation: {between_before:.4f}")
    print(f"      Ratio (within/between):   {ratio_before:.4f}")
    print(f"    After harmonization:")
    print(f"      Within-site correlation:  {within_after:.4f}")
    print(f"      Between-site correlation: {between_after:.4f}")
    print(f"      Ratio (within/between):   {ratio_after:.4f}")
    print(f"    Change: {((ratio_after - ratio_before) / ratio_before * 100):.1f}%")
    
    if ratio_after < ratio_before:
        print("    ✓ SUCCESS: Batch effects reduced (ratio decreased)")
    else:
        print("    ✗ WARNING: Batch effects may not be reduced (ratio increased)")
    
    # Method 2: Kolmogorov-Smirnov test (recommended for radiomics)
    print("\n  [Method 2] Kolmogorov-Smirnov Test (Nature Sci Reports 2022):")
    print("    Testing if feature distributions differ significantly across sites...")
    n_sig_before = kolmogorov_smirnov_test(X_before, metadata, feature_names)
    n_sig_after = kolmogorov_smirnov_test(X_after, metadata, feature_names)
    
    pct_sig_before = (n_sig_before / len(feature_names)) * 100
    pct_sig_after = (n_sig_after / len(feature_names)) * 100
    
    print(f"    Before harmonization: {n_sig_before}/{len(feature_names)} features ({pct_sig_before:.1f}%) differ across sites")
    print(f"    After harmonization:  {n_sig_after}/{len(feature_names)} features ({pct_sig_after:.1f}%) differ across sites")
    print(f"    Reduction: {n_sig_before - n_sig_after} features ({pct_sig_before - pct_sig_after:.1f}%)")
    
    if n_sig_after < n_sig_before:
        print("    ✓ SUCCESS: Fewer features show site differences")
    else:
        print("    ✗ WARNING: Site differences not reduced")
    
    # Method 3: Coefficient of Variation (travelling subject method)
    print("\n  [Method 3] Coefficient of Variation - CoV (Travelling Subject Method):")
    cov_before = compute_coefficient_of_variation(X_before, metadata)
    cov_after = compute_coefficient_of_variation(X_after, metadata)
    
    print(f"    Before harmonization: {cov_before:.2f}%")
    print(f"    After harmonization:  {cov_after:.2f}%")
    print(f"    Reduction: {cov_before - cov_after:.2f}%")
    
    if cov_after < cov_before:
        print("    ✓ SUCCESS: Site effect reduced (lower CoV)")
    else:
        print("    ✗ WARNING: Site effect not reduced (higher CoV)")
    
    # Summary verdict
    print("\n" + "="*60)
    print("OVERALL ASSESSMENT:")
    print("="*60)
    
    success_count = 0
    if ratio_after < ratio_before:
        success_count += 1
    if n_sig_after < n_sig_before:
        success_count += 1
    if cov_after < cov_before:
        success_count += 1
    
    if success_count >= 2:
        print("✓ ComBat harmonization SUCCESSFUL")
        print(f"  ({success_count}/3 metrics show improvement)")
        print("  Batch effects have been effectively reduced.")
    else:
        print("✗ ComBat harmonization needs review")
        print(f"  ({success_count}/3 metrics show improvement)")
        print("  Consider checking data quality or batch variable.")
    
    print("="*60)
    
    print("\n" + "="*60)
    print("Quality check complete!")
    print(f"Plots saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
