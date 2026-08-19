# charts with the rosenholtz saliency metrics
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# =========================================================================
# CONFIGURATION
# =========================================================================
INPUT_CSV = r"results\newRosenholtzSaliency15mm.csv"
SALIENCY_COLS = ['Local_Nodule_Saliency', 'Local_Perinodular_Saliency', 'Local_Total_Mask_Saliency']
COLUMN_ALIASES = {
    'Local_Nodule_Saliency': ['Local_Nodule_Saliency', 'Local_Nodule_Clutter'],
    'Local_Perinodular_Saliency': ['Local_Perinodular_Saliency', 'Local_Perinodular_Clutter'],
    'Local_Total_Mask_Saliency': ['Local_Total_Mask_Saliency', 'Local_Total_Mask_Clutter'],
}

# Image files to exclude from the analysis (matched against the Image_ID column).
# Add IDs here to drop them, e.g. EXCLUDE_IMAGES = ["JPCLN084.png", "JPCLN106.png"].
# Matching ignores the file extension AND any spaces, so "JPCLN 084",
# "JPCLN084", and "JPCLN084.png" all match the same image.
EXCLUDE_IMAGES = [
    'JPCLN119', 'JPCLN133', 'JPCLN138', 'JPCLN142', 'JPCLN143',
    'JPCLN144', 'JPCLN145', 'JPCLN146', 'JPCLN153', 'JPCLN154',
]

APPLY_P_CORRECTIONS = True  # Apply multiple-comparison p-value correction
#P_CORRECTION_METHOD = "BH"  # Options: "bonferroni" or "BH"

# =========================================================================
# 1. LOAD & CLEAN DATA
# =========================================================================
print("Loading and cleaning data...")
df = pd.read_csv(INPUT_CSV)

# Drop any excluded image files. Matching drops the extension and any spaces,
# so "JPCLN 119", "JPCLN119", and "JPCLN119.png" all match.
if EXCLUDE_IMAGES and 'Image_ID' in df.columns:
    def _norm(s):
        return os.path.splitext(str(s))[0].replace(" ", "").upper()
    exclude_stems = {_norm(e) for e in EXCLUDE_IMAGES}
    drop_mask = df['Image_ID'].map(_norm).isin(exclude_stems)
    if drop_mask.any():
        print(f"Excluding {int(drop_mask.sum())} image(s): "
              f"{', '.join(df.loc[drop_mask, 'Image_ID'].astype(str))}")
    df = df[~drop_mask]

for canonical_col, aliases in COLUMN_ALIASES.items():
    matched_col = next((alias for alias in aliases if alias in df.columns), None)
    if matched_col is None:
        raise KeyError(f"Could not find any column for {canonical_col}. Available columns: {list(df.columns)}")
    if matched_col != canonical_col:
        df[canonical_col] = df[matched_col]

df = df.dropna(subset=['Nodule_Rating'] + SALIENCY_COLS)
print(f"✅ Loaded {len(df)} rows smoothly.\n")


def _correct_p_values(p_values):
    """Return Bonferroni and BH-FDR corrected p-values for a list of p-values."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return [], []

    bonferroni = np.minimum(p * m, 1.0)

    order = np.argsort(p)
    ranks = np.arange(1, m + 1)
    bh = np.empty(m, dtype=float)
    bh[order] = np.minimum.accumulate((p[order] * m / ranks)[::-1])[::-1]
    bh = np.minimum(bh, 1.0)

    return bonferroni, bh


# =========================================================================
# 2. STATISTICAL ANALYSIS (Correlations & ANOVA)
# =========================================================================
print("="*40 + "\n📈 SALIENCY STATISTICAL ANALYSIS\n" + "="*40)

results = []
for col in SALIENCY_COLS:
    p_corr, p_val = pearsonr(df['Nodule_Rating'], df[col])
    s_corr, s_val = spearmanr(df['Nodule_Rating'], df[col])
    results.append({
        'metric': col,
        'pearson_r': p_corr,
        'pearson_p': p_val,
        'spearman_r': s_corr,
        'spearman_p': s_val,
    })

if APPLY_P_CORRECTIONS and results:
    spearman_p_values = [row['spearman_p'] for row in results]
    bonferroni_p, bh_p = _correct_p_values(spearman_p_values)
    for row, b_p, fdr_p in zip(results, bonferroni_p, bh_p):
        row['bonferroni_p'] = b_p
        row['bh_fdr_p'] = fdr_p

for row in results:
    signif = "✅ Significant" if row['pearson_p'] < 0.05 else "❌ Not Significant"
    print(f"{row['metric']:<25}")
    print(f"  -> Pearson  r: {row['pearson_r']:.3f} (p-value: {row['pearson_p']:.4f})")
    print(f"  -> Spearman r: {row['spearman_r']:.3f} (p-value: {row['spearman_p']:.4f})")
    if APPLY_P_CORRECTIONS:
        print(f"  -> Bonferroni p: {row['bonferroni_p']:.4f}")
        print(f"  -> BH-FDR p: {row['bh_fdr_p']:.4f}")
    print(f"  -> Status: {signif}\n")

# =========================================================================
# 3. VISUALIZATION (Automated Multi-Plot Grid Dashboard)
# =========================================================================
print("📊 Generating the multi-metric saliency visualization dashboard...")

# Create a 1x3 grid of plots for the three local saliency metrics
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Flatten the 1x3 grid matrix into a 1D list [0, 1, 2]
axes = axes.flatten()

# Explicitly use clean, ordered ratings for the X-axis
ratings_order = [1, 2, 3, 4, 5]

# Loop through each saliency metric and assign it its own boxplot panel
for i, col in enumerate(SALIENCY_COLS):
    # Group the specific clutter metric's values by the rating order
    grouped_data = [df[df['Nodule_Rating'] == r][col].values for r in ratings_order]
    
    # Target the specific panel in the grid axes[i]
    axes[i].boxplot(grouped_data, label=ratings_order)
    
    # Add clear titles and clean up string labels for display
    clean_title = col.replace('_', ' ')
    axes[i].set_title(f"{clean_title} vs Rating", fontsize=11, fontweight='bold')
    axes[i].set_xlabel("Nodule Rating (1=Hard, 5=Easy)", fontsize=9)
    axes[i].set_ylabel("Saliency Score", fontsize=9)
    axes[i].grid(True, linestyle='--', alpha=0.5)

# Keep layout spacing perfectly neat and prevent text overlaps
plt.tight_layout()

# Save the comprehensive dashboard asset
outputplot = r"C:\Users\vivia\OneDrive\Desktop\lab\new_saliency_metrics2115mm.png"
plt.savefig(outputplot, dpi=300, bbox_inches='tight')

print(f"🎉 Dashboard complete! Saved to {outputplot}")
plt.show()