import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

df = pd.read_csv('nodule_brightness_resultsextra.csv')

# Calculate both correlations
pearson_corr, pearson_p = pearsonr(df['average_brightness'], df['clarity'])
spearman_corr, spearman_p = spearmanr(df['average_brightness'], df['clarity'])

print(f"Pearson Correlation: {pearson_corr:.3f} (p-value: {pearson_p:.6f})")
print(f"Spearman Correlation: {spearman_corr:.3f} (p-value: {spearman_p:.6f})")

# Visualize
plt.scatter(df['clarity'], df['average_brightness'], alpha=0.6)
plt.xlabel('Clarity Rating')
plt.ylabel('Average Brightness')
plt.title(f'Clarity vs Brightness (Pearson r={pearson_corr:.3f})')
plt.grid(True)
plt.show()