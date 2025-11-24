# UTMF-CRISP  
**Unified Temporal–Measurement Framework (UTMF)**  
Empirical detection of universal multifractal coherence across 60 orders of magnitude — from quantum random bits to cosmic microwave background.

**Author**: Jedi Markus Strive  
**Contact**: crisplatform@gmail.com, x.com/JediStrive19156  
**License**: MIT  
**arXiv preprint** (coming soon): [link invullen zodra geüpload]

## What is UTMF?
A minimal, fully reproducible pipeline that:
1. Takes raw scientific datasets (LIGO, Planck, DESI, CERN, NIST, QRNG, Gaia, …)
2. Runs a robust custom MFDFA (multifractal detrended fluctuation analysis)
3. Extracts h(q) multifractal spectra
4. Computes **TCI**, **MCI** and **TMCI** — three coherence indices
5. Consistently finds TMCI ≈ 0.92–0.94 across radically different physical domains

The signal survives:
- no denoising  
- higher detrending  
- narrow q-range  
- shuffled data (drops to ~0.05–0.15)

→ Strongly suggests a real universal pattern, not an artifact.

## Quick start (Google Colab)
1. Open the notebook: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jedi-Markus-Strive/UTMF-CRISP/blob/main/UTMF_Main.ipynb)
2. Mount your Google Drive (datasets must be in `/MyDrive/Datasets_UTMF/`)
3. Run all cells → you will reproduce TMCI ≈ 0.93 in ~15–20 minutes

## Repository structure

UTMF-CRISP/
├── UTMF_Main.ipynb              ← complete pipeline (Cell 1 + Cell 2)
├── config.py                     ← all parameters and paths
├── utils.py                      ← helper functions
├── outputs/                      ← example CSV + full JSON (latest run)
├── README.md
└── LICENSE





