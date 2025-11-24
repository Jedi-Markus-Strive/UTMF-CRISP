# UTMF-CRISP

**Unified Temporal–Measurement Framework (UTMF)**  
Empirical evidence for cross-domain multifractal coherence in high-precision physical measurements.

**Author**: Jedi Markus Strive  
**Contact**: http://x.com/JediStrive19156, crisplatform@gmail.com 
**License**: MIT  
**arXiv preprint**: [coming soon – link will be added upon upload]

## Overview

UTMF is a minimal, fully reproducible, domain-agnostic pipeline that applies multifractal detrended fluctuation analysis (MFDFA) to heterogeneous scientific datasets and quantifies coherence of the resulting h(q) spectra via three indices (TCI, MCI, TMCI).

Tested datasets (19 total):
- Quantum random number generator (NIST)
- Gravitational-wave strain (LIGO-L1)
- Cosmic microwave background (Planck)
- Galaxy photometry (DESI)
- Atomic transition spectra (NIST)
- Collider events (CERN)
- Pulsar timing residuals (NANOGrav 15 yr)
- Stellar astrometry (Gaia DR3)

Key result: real datasets consistently yield high coherence (MCI = 1.026 ± 0.194), while 18 synthetic null models occupy a distinct low-coherence regime (MCI = 0.276 ± 0.157). Separation is decisive across 13 independent statistical tests.

## Repository contents

- `UTMF_main.py` – complete analysis pipeline  
- `validation_synthetic_controls.py` – real vs synthetic null-model comparison (generates Figure 4)  
- `final_statistical_validation.py` – 13 statistical tests  
- `prepare_NANOGrav_15yr_data.ipynb` – extracts the official NANOGrav 15-yr dataset  
- `Datasets/` – target directory for all raw and extracted data  
- `UTMF_outputs/` – automatically generated CSVs, figures and JSON artefacts  
- Full 47.3 MB JSON artefact (all individual MFDFA results): permanently archived on Zenodo  
  DOI: [10.5281/zenodo.17698176](https://doi.org/10.5281/zenodo.17698176)

## Quick start (Google Colab)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jedi-Markus-Strive/UTMF-CRISP/blob/main/UTMF_main.py)

1. Place required datasets in a folder `Datasets_UTMF` on your Google Drive  
2. Run `prepare_NANOGrav_15yr_data.ipynb` first if using pulsar data  
3. Execute the notebooks in order  
→ Full reproduction including TMCI ≈ 0.93 and complete null-model separation in ~20–30 minutes.

## Citation

When the preprint is live, please cite the arXiv identifier.  
The permanent data archive should be cited as:

> Jedi Markus Strive (2025). UTMF-CRISP – Full MFDFA results and artefacts (2025-11-24 run) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17698176
