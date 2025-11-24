# UTMF — Validation: Real datasets vs Synthetic controls
# Author: Jedi Markus Strive
# Generates timestamped summary CSV and Figure 4 (as used in the preprint)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ------------------------------------------------------------------
timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
print(f"UTMF synthetic control validation — {timestamp}\n")

# ------------------------------------------------------------------
# 1. Load real dataset results (assumes results_all from UTMF_main.ipynb is in memory)
real_mci = []
real_names = []

for name, item in results_all.items():
    res = item[0] if isinstance(item, tuple) else item
    if not isinstance(res, dict) or 'hq_values' not in res:
        continue
    hq_list = res['hq_values']
    if not hq_list:
        continue
    mean_hq = np.nanmean(hq_list, axis=0)
    real_mci.append(mean_hq)
    real_names.append(name)

print(f"Valid real datasets: {len(real_mci)}")

# ------------------------------------------------------------------
# 2. Synthetic data generation
def generate_synthetic(template, kind, seed=42):
    rng = np.random.default_rng(seed)
    n = len(template)
    if kind == "white":
        return rng.normal(0, 1, n)
    if kind == "ar1":
        out = np.zeros(n); out[0] = template[0]
        for i in range(1, n):
            out[i] = 0.8 * out[i-1] + rng.normal(0, 1)
        return out
    if kind == "pink":
        w = rng.normal(0, 1, n)
        fft = np.fft.rfft(w)
        k = np.fft.rfftfreq(n); k[0] = 1
        fft /= np.sqrt(k)
        return np.fft.irfft(fft, n=n)
    if kind == "fgn":  # H ≈ 0.8
        f = np.fft.rfftfreq(n); f[0] = 1
        psd = np.abs(f) ** (-0.8)
        w = rng.normal(0, 1, len(f)) + 1j*rng.normal(0, 1, len(f))
        return np.fft.irfft(w * np.sqrt(psd), n=n)
    if kind == "phase_surrogate":
        fft = np.fft.rfft(template)
        mag = np.abs(fft)
        phase = rng.uniform(0, 2*np.pi, len(fft))
        return np.fft.irfft(mag * np.exp(1j*phase), n=n)
    if kind == "mock_ligo":
        t = np.arange(n)
        noise = rng.normal(0, 1, n)
        chirp = np.sin(2*np.pi*50*np.sqrt(t/n)) * np.exp(-((t-n//2)/300)**2)
        return noise + 2*chirp

# Use first real mean h(q) as length template
template = real_mci[0]
q_values = CONFIG['mfdfa']['q_values']
scales   = np.logspace(np.log10(8), np.log10(len(template)//8), 15).astype(int)

synth_kinds = ["white", "ar1", "pink", "fgn", "phase_surrogate", "mock_ligo"]
synth_mci = []
synth_names = []

for kind in synth_kinds:
    for rep in range(3):
        series = generate_synthetic(template, kind, seed=rep)
        _, hq, _, _, _ = jedi_mfdfa(series, scales, q_values, detrend_order=2)
        hq = np.where(np.isfinite(hq), hq, 0.5)
        synth_mci.append(hq)
        synth_names.append(f"SYN_{kind}_{rep+1}")

print(f"Valid synthetic datasets: {len(synth_mci)}\n")

# ------------------------------------------------------------------
# 3. Combine and save CSV
df = pd.DataFrame({
    "dataset": real_names + synth_names,
    "label":   ["real"]*len(real_mci) + ["synth"]*len(synth_mci),
    "local_mci": [np.nanmean(hq) for hq in (real_mci + synth_mci)]
})

csv_path = f"UTMF_outputs/UTMF_synthetic_validation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
os.makedirs("UTMF_outputs", exist_ok=True)
df.to_csv(csv_path, index=False, float_format="%.6f")
print(f"Summary CSV saved → {csv_path}")

# ------------------------------------------------------------------
# 4. Figure 4
plt.figure(figsize=(10, 6.5))
sns.violinplot(data=df, x="label", y="local_mci", hue="label",
               palette=["#1f77b4", "#d62728"], inner="quartile", legend=False)
sns.stripplot(data=df, x="label", y="local_mci", hue="label",
              palette=["#1f77b4", "#d62728"], size=9, jitter=True, alpha=0.9,
              edgecolor="black", linewidth=1.2, legend=False)

plt.title("UTMF Null-Model Validation\nReal datasets vs Synthetic controls", fontsize=16, pad=20)
plt.ylabel("Measurement Coherence Index (MCI)")
plt.xlabel("")
plt.ylim(-0.05, 1.35)
plt.grid(axis='y', alpha=0.35)

real_vals  = df[df['label']=='real']['local_mci']
synth_vals = df[df['label']=='synth']['local_mci']
text = (f"Real datasets (n={len(real_vals)})\n"
        f"MCI = {real_vals.mean():.3f} ± {real_vals.std():.3f}\n\n"
        f"Synthetic controls (n={len(synth_vals)})\n"
        f"MCI = {synth_vals.mean():.3f} ± {synth_vals.std():.3f}")

plt.text(0.75, 0.96, text, transform=plt.gca().transAxes, fontsize=9.5,
         va='top', ha='left', bbox=dict(facecolor="white", alpha=0.96, edgecolor="black"))

plt.tight_layout()
fig_path = f"UTMF_outputs/UTMF_Figure4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
plt.savefig(fig_path, dpi=400, bbox_inches='tight')
plt.show()
print(f"Figure saved → {fig_path}")
