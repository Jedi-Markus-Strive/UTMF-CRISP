[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17698176.svg)](https://doi.org/10.5281/zenodo.17698176)


# UTMF-CRISP

**Unified Temporal–Measurement Framework (UTMF)**  
Empirical evidence for cross-domain multifractal coherence in high-precision physical measurements.

**Author**: Jedi Markus Strive  
**Contact**: http://x.com/JediStrive19156, crisplatform@gmail.com 
**License**: MIT  
**arXiv preprint**: [coming soon – link will be added upon upload]

## Overview

UTMF is a minimal, fully reproducible, domain-agnostic pipeline that applies multifractal detrended fluctuation analysis (MFDFA) to heterogeneous scientific datasets and quantifies coherence of the resulting h(q) spectra via three indices: TCI (Temporal Coherence Index), MCI (Measurement Coherence Index), TMCI (Temporal-Measurement Coherence Index).

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

- `UTMF_v5.0_Main.py` – complete analysis pipeline  
- `validation_synthetic_controls.py` – real vs synthetic null-model comparison   
- `final_statistical_validation.py` – 13 statistical tests  (generates Figure 4)
- `prepare_NANOGrav_15yr_data.ipynb` – extracts the official NANOGrav 15-yr dataset  
- `Datasets/` – target directory for all raw and extracted data  
- `UTMF_outputs/` – automatically generated CSVs, figures and JSON artefacts  
- Full 47.3 MB JSON artefact (all individual MFDFA results): permanently archived on Zenodo  
  DOI: [10.5281/zenodo.17698176](https://doi.org/10.5281/zenodo.17698176)

## Quick start (Google Colab)

[![Open UTMF Launcher in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jedi-Markus-Strive/UTMF-CRISP/blob/main/UTMF_launcher.ipynb)

1. Click the badge above → opens the launcher notebook in Colab
2. Mount your Google Drive (prompt appears) – place datasets in `/MyDrive/Datasets_UTMF/`
3. Run all cells → clones repo, runs all .py scripts, extracts NANOGrav, generates outputs
→ Full reproduction in ~20–30 minutes (TMCI ≈ 0.93 + null-model separation)

## Citation

When the preprint is live, please cite the arXiv identifier.  
The permanent data archive should be cited as:

> Jedi Markus Strive (2025). UTMF-CRISP – Full MFDFA results and artefacts (2025-11-24 run) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17698176
