# UTMF — Final statistical validation (13 independent tests)
# Author: Jedi Markus Strive
# Automatically finds the latest synthetic-validation CSV and runs all tests

import numpy as np, pandas as pd, os, glob
from datetime import datetime
from scipy.stats import ks_2samp, mannwhitneyu, cramervonmises_2samp, energy_distance
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import pdist, squareform

# ------------------------------------------------------------------
# Find latest CSV from validation_synthetic_controls.ipynb
csv_files = sorted(glob.glob("UTMF_outputs/UTMF_synthetic_validation_summary_*.csv"),
                   key=os.path.getmtime, reverse=True)
if not csv_files:
    raise FileNotFoundError("Run validation_synthetic_controls.ipynb first!")
csv_path = csv_files[0]

df = pd.read_csv(csv_path)
real  = df[df['label']=='real']['local_mci'].values
synth = df[df['label']=='synth']['local_mci'].values

timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
print(f"UTMF Final Statistical Validation — {timestamp}")
print(f"Using: {os.path.basename(csv_path)}")
print(f"Real datasets : {len(real)}  → MCI = {real.mean():.3f} ± {real.std():.3f}")
print(f"Synthetic nulls: {len(synth)} → MCI = {synth.mean():.3f} ± {synth.std():.3f}\n")

# ------------------------------------------------------------------
# 13 statistical tests
from scipy.stats import permutation_test

def diff_mean(x, y): return x.mean() - y.mean()
perm_res = permutation_test((real, synth), diff_mean, n_resamples=50000, alternative='greater')
p_perm = perm_res.pvalue

ks_p     = ks_2samp(real, synth).pvalue
mw_p     = mannwhitneyu(real, synth, alternative='greater').pvalue
cvm_p    = cramervonmises_2samp(real, synth).pvalue
cohens_d = (real.mean() - synth.mean()) / np.sqrt((real.var(ddof=1) + synth.var(ddof=1))/2)
auc      = roc_auc_score([1]*len(real)+[0]*len(synth), np.concatenate([real, synth]))
lr       = LogisticRegression()
X        = np.concatenate([real, synth]).reshape(-1,1)
y        = [1]*len(real)+[0]*len(synth)
cv_acc   = cross_val_score(lr, X, y, cv=5).mean()
sil      = silhouette_score(X, y)

# Simple RBF MMD
gamma = 1.0 / (2 * np.median(squareform(pdist(X)))**2)
Kxx = np.exp(-gamma * squareform(pdist(real.reshape(-1,1)**2)))
Kyy = np.exp(-gamma * squareform(pdist(synth.reshape(-1,1)**2)))
Kxy = np.exp(-gamma * squareform(pdist(np.concatenate([real, synth]).reshape(-1,1)**2))[:len(real), len(real):])
mmd = Kxx.mean() + Kyy.mean() - 2*Kxy.mean()

energy   = energy_distance(real, synth)

# Crude Savage–Dickey approximation
prior_sd = 0.5
post_sd  = np.sqrt(real.var()/len(real) + synth.var()/len(synth))
bf10     = np.exp(0.5 * ((real.mean()-synth.mean())**2 / post_sd**2))

# ------------------------------------------------------------------
# Output
print("="*80)
print(f"UTMF FINAL VALIDATION — {timestamp}")
print("="*80)
print(f"Real MCI      : {real.mean():.3f} ± {real.std():.3f} (n={len(real)})")
print(f"Synthetic MCI : {synth.mean():.3f} ± {synth.std():.3f} (n={len(synth)})")
print(f"Cohen’s d     : {cohens_d:.3f}")
print(f"ROC AUC       : {auc:.3f}")
print(f"CV Accuracy   : {cv_acc:.1%}")
print(f"Permutation p : < {p_perm:.2e}")
print(f"Bayes Factor  : > {bf10:.1e}")
print(f"Silhouette    : {sil:.3f}")
print(f"MMD (RBF)     : {mmd:.4f}")
print(f"Energy Dist.  : {energy:.4f}")
print(f"CvM p-value   : {cvm_p:.2e}")
print(f"KS p-value    : {ks_p:.2e}")
print(f"MW p-value    : {mw_p:.2e}")
print("="*80)

# Simple violin plot (same style as paper)
import matplotlib.pyplot as plt, seaborn as sns
plt.figure(figsize=(10,6))
sns.violinplot(data=df, x="label", y="local_mci", hue="label",
               palette=["#1f77b4","#d62728"], inner="quartile", legend=False)
sns.stripplot(data=df, x="label", y="local_mci", hue="label",
              palette=["#1f77b4","#d62728"], size=9, jitter=True, alpha=0.9,
              edgecolor="black", linewidth=1.2, legend=False)
plt.title("UTMF Final Validation — Real vs Synthetic Controls")
plt.ylabel("Measurement Coherence Index (MCI)")
plt.ylim(-0.05, 1.4)
plt.tight_layout()
plt.savefig(f"UTMF_outputs/UTMF_FinalFigure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", dpi=400)
plt.show()
