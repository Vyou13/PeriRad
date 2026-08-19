import os
import pandas as pd
import numpy as np
from scipy import stats

# =========================================================================
# CORRELATION: Nodule_Rating vs. subband-entropy clutter scores
# =========================================================================
# For each clutter column we report:
#   - Pearson r  (linear association)
#   - Spearman rho (rank/monotonic association -- the right test for an
#     ordinal 1-5 rating scale)
# together with two-sided p-values, plus the mean clutter per rating so you
# can see the trend directly.
# =========================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Input clutter CSVs produced by subband_entropy_clutter.py
INPUT_FILES = {
    "10mm_size10": os.path.join(RESULTS_DIR, "subbandEntropy10mm_10-10mm.csv"),
    "10mm_size15": os.path.join(RESULTS_DIR, "subbandEntropy10mm_15-15mm.csv"),
    "10mm_size20": os.path.join(RESULTS_DIR, "subbandEntropy10mm_20-20mm.csv"),
}

CLUTTER_COLUMNS = [
    "Local_Nodule_Clutter",
    "Local_Perinodular_Clutter",
    "Local_Total_Mask_Clutter",
]

RATING_COL = "Nodule_Rating"

# Image files to exclude from the analysis (matched against the Image_ID column).
# Add IDs here to drop them, e.g. EXCLUDE_IMAGES = ["JPCLN084.png", "JPCLN106.png"].
# Matching ignores the file extension AND any spaces, so "JPCLN 084",
# "JPCLN084", and "JPCLN084.png" all match the same image.
EXCLUDE_IMAGES = [
    'JPCLN119', 'JPCLN133', 'JPCLN138', 'JPCLN142', 'JPCLN143',
    'JPCLN144', 'JPCLN145', 'JPCLN146', 'JPCLN153', 'JPCLN154',
]

# Optionally save plots (needs matplotlib); set to False to skip.
MAKE_PLOTS = True

# Plot layout:
#   True  -> one combined 2x3 figure per dataset (box plots on top row,
#            scatter plots on bottom row) = 1 file per dataset.
#   False -> two separate 1x3 figures per dataset (box + scatter) = 2 files.
COMBINED_2x3 = True

OUTPUT_CSV = os.path.join(RESULTS_DIR, "rating_clutter_correlationfilter.csv")


def analyze(label, csv_path):
    """Compute correlations for one input file; return a list of summary rows."""
    if not os.path.exists(csv_path):
        print(f"[{label}] File not found, skipping: {csv_path}")
        return []

    df = pd.read_csv(csv_path)

    # Drop any excluded image files. Matching drops the extension and any
    # spaces, so "JPCLN 119", "JPCLN119", and "JPCLN119.png" all match.
    if EXCLUDE_IMAGES and "Image_ID" in df.columns:
        def _norm(s):
            return os.path.splitext(str(s))[0].replace(" ", "").upper()
        exclude_stems = {_norm(e) for e in EXCLUDE_IMAGES}
        stems = df["Image_ID"].map(_norm)
        drop_mask = stems.isin(exclude_stems)
        if drop_mask.any():
            print(f"[{label}] Excluding {int(drop_mask.sum())} image(s): "
                  f"{', '.join(df.loc[drop_mask, 'Image_ID'].astype(str))}")
        df = df[~drop_mask]

    print("\n" + "=" * 70)
    print(f"[{label}] {os.path.basename(csv_path)}  ({len(df)} rows)")
    print("=" * 70)

    rows = []
    for col in CLUTTER_COLUMNS:
        if col not in df.columns:
            print(f"  Column missing: {col}")
            continue

        # Drop rows where either the rating or the clutter value is missing.
        pair = df[[RATING_COL, col]].dropna()
        n = len(pair)
        if n < 3:
            print(f"  {col}: not enough data (n={n})")
            continue

        x = pair[RATING_COL].astype(float)
        y = pair[col].astype(float)

        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)

        signif = "✅ Significant" if spearman_p < 0.05 else "❌ Not Significant"
        direction = "positive" if spearman_r > 0 else "negative"

        clean_metric = col.replace('_', ' ')
        print(f"\n{clean_metric}")
        print(f"  -> Pearson  r: {pearson_r:+.3f} (p-value: {pearson_p:.4f})")
        print(f"  -> Spearman r: {spearman_r:+.3f} (p-value: {spearman_p:.4f})")
        print(f"  -> Status: {signif} ({direction} correlation)")

        # Mean clutter per rating -- shows the direction of the trend.
        means = pair.groupby(RATING_COL)[col].mean()
        trend = "  ".join(f"r{int(r)}={m:.2f}" for r, m in means.items())
        print(f"  -> Mean by rating: {trend}")

        rows.append({
            "Dataset": label,
            "Clutter_Metric": col,
            "N": n,
            "Pearson_r": round(pearson_r, 4),
            "Pearson_p": round(pearson_p, 5),
            "Spearman_rho": round(spearman_r, 4),
            "Spearman_p": round(spearman_p, 5),
            "Significant_p<0.05": "yes" if spearman_p < 0.05 else "no",
        })

    if MAKE_PLOTS:
        _dashboard(label, df, rows)

    return rows


def _draw_box(ax, col, df, stat):
    """Draw one box-plot panel (clutter by rating) on the given axis."""
    ratings_order = [1, 2, 3, 4, 5]
    grouped = [df[df[RATING_COL] == r][col].dropna().values for r in ratings_order]
    bp = ax.boxplot(grouped, labels=ratings_order, showmeans=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#9ecae1")
        patch.set_alpha(0.7)
    # jittered raw points for context
    for j, g in enumerate(grouped, start=1):
        jitter = (np.arange(len(g)) % 7 - 3) * 0.03
        ax.scatter(np.full(len(g), j) + jitter, g, s=10, color="#08519c", alpha=0.4)
    _annotate(ax, col, stat)


def _draw_scatter(ax, col, df, stat):
    """Draw one scatter panel (clutter vs rating) on the given axis."""
    pair = df[[RATING_COL, col]].dropna()
    x = pair[RATING_COL].astype(float)
    y = pair[col].astype(float)
    jitter = (np.arange(len(x)) % 7 - 3) * 0.02
    ax.scatter(x + jitter, y, alpha=0.5, s=18, label="nodules")

    means = pair.groupby(RATING_COL)[col].mean()
    ax.plot(means.index, means.values, "o-", color="crimson", label="mean per rating")

    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        xr = np.array([x.min(), x.max()])
        ax.plot(xr, slope * xr + intercept, "--", color="gray", label="linear fit")
    _annotate(ax, col, stat)
    ax.legend(fontsize=8)


def _annotate(ax, col, stat):
    rho = stat.get("Spearman_rho", float("nan")) if stat else float("nan")
    p = stat.get("Spearman_p", float("nan")) if stat else float("nan")
    ax.set_title(f"{col.replace('_', ' ')} vs Rating\n(Spearman rho={rho:+.3f}, p={p:.3g})",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Nodule Rating (1=Hard, 5=Easy)", fontsize=9)
    ax.set_ylabel("Clutter Score", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)


def _dashboard(label, df, rows):
    """Build the plot dashboard(s) for one dataset. Layout controlled by
    COMBINED_2x3: True -> single 2x3 figure; False -> two 1x3 figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless / no display needed
        import matplotlib.pyplot as plt
    except ImportError:
        return  # matplotlib not installed -> silently skip plots

    stats_by_col = {r["Clutter_Metric"]: r for r in rows}

    if COMBINED_2x3:
        # One figure: box plots on top row, scatter plots on bottom row.
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        for i, col in enumerate(CLUTTER_COLUMNS):
            stat = stats_by_col.get(col)
            for ax in (axes[0, i], axes[1, i]):
                if col not in df.columns:
                    ax.set_visible(False)
            if col not in df.columns:
                continue
            _draw_box(axes[0, i], col, df, stat)
            _draw_scatter(axes[1, i], col, df, stat)
        fig.suptitle(f"{label}: Clutter vs nodule rating (box + scatter)",
                     fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_png = os.path.join(RESULTS_DIR, f"filterclutter_dashboard_{label}.png")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  2x3 dashboard saved: {out_png}")
        return

    # Two separate 1x3 figures.
    for kind, drawer in (("boxplots", _draw_box), ("scatter", _draw_scatter)):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes = axes.flatten()
        for i, col in enumerate(CLUTTER_COLUMNS):
            if col not in df.columns:
                axes[i].set_visible(False)
                continue
            drawer(axes[i], col, df, stats_by_col.get(col))
        fig.suptitle(f"{label}: Clutter vs nodule rating ({kind})",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_png = os.path.join(RESULTS_DIR, f"clutter_{kind}_{label}.png")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  1x3 {kind} dashboard saved: {out_png}")


def _apply_corrections(summary):
    """Add Bonferroni and Benjamini-Hochberg (FDR) columns for the Spearman p-values."""
    p = summary["Spearman_p"].to_numpy(dtype=float)
    m = len(p)

    # Bonferroni: p * m, capped at 1.
    summary["Bonferroni_p"] = np.minimum(p * m, 1.0).round(5)

    # Benjamini-Hochberg: rank p ascending, adjust, then enforce monotonicity.
    order = np.argsort(p)
    ranks = np.arange(1, m + 1)
    bh = np.empty(m)
    bh[order] = np.minimum.accumulate((p[order] * m / ranks)[::-1])[::-1]
    summary["BH_FDR_p"] = np.minimum(bh, 1.0).round(5)

    summary["Significant_Bonferroni"] = np.where(summary["Bonferroni_p"] < 0.05, "yes", "no")
    summary["Significant_BH_FDR"] = np.where(summary["BH_FDR_p"] < 0.05, "yes", "no")
    return summary


def main():
    all_rows = []
    for label, path in INPUT_FILES.items():
        all_rows.extend(analyze(label, path))

    if not all_rows:
        print("\nNo results computed.")
        return

    summary = _apply_corrections(pd.DataFrame(all_rows))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))
    print(f"\nSaved correlation summary to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
