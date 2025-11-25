# UTMF 5.0 Configuration - (Cell 1)
# Author: Jedi Markus Strive
# Version: 5.0
# Date: November 24, 2025

# This cell handles all data loading, preprocessing, initial fractal analysis (MFDFA), and optional index computations (TCI/MCI/TMCI) for UTMF.
# Why: Reproducible subsets from diverse datasets; compute D_f, TMCI.
# 1. Mount Drive & set paths.
# 2. Load configs; toggle datasets/indices via 'utmf_use'/'compute_indices'.
# 3. Load → Denoise → MFDFA → (if enabled) TCI/MCI/TMCI.
# 4. Output: Globals + metadata CSV with all metrics (incl. cross-val if enabled). Optional full JSON.
# Run order: Cell 1 first; globals for later cells.
# Install required packages (run once per session)

!pip install healpy joblib h5py uproot pywavelets pint-pulsar numpy pandas scipy matplotlib tqdm dtaidistance tensorly dask tensorflow torch plotly gudhi ripser scikit-learn seaborn pytz networkx MFDFA

# Import libraries
import networkx as nx
import numpy as np
import pandas as pd
import h5py
from astropy.io import fits
import healpy as hp
import scipy.signal
import dask.array as da
import gc
from tqdm import tqdm
from numba import jit
import uproot
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold  # Voor cross-val
from scipy.stats import pearsonr, norm, entropy
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from google.colab import drive
from joblib import Parallel, delayed
import os
import glob
import warnings
import json  # For serializing nested metadata to CSV
from datetime import datetime
import pytz
import requests
import time
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")
try:
    from pint.models import get_model, get_model_and_toas
    from pint.toa import get_TOAs
except ImportError:
    print("Error: pint-pulsar not installed. Install with `pip install pint-pulsar`.")

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Output directory for CSVs
OUTPUT_DIR = '/content/drive/MyDrive/Datasets_UTMF/UTMF_outputs/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output dir: {OUTPUT_DIR}")

# Element names (for NIST_3 dataset)
ELEMENT_NAMES = {
    'Ac': 'Actinium', 'Ag': 'Silver', 'Al': 'Aluminium', 'Ar': 'Argon', 'B': 'Boron',
    'Ba': 'Barium', 'Be': 'Beryllium', 'Bi': 'Bismuth', 'Br': 'Bromine', 'C': 'Carbon',
    'Ca': 'Calcium', 'Cd': 'Cadmium', 'Ce': 'Cerium', 'Cl': 'Chlorine', 'Co': 'Cobalt',
    'Cr': 'Chromium', 'Cs': 'Cesium', 'Cu': 'Copper', 'Dy': 'Dysprosium', 'Eu': 'Europium',
    'F': 'Fluorine', 'Fe': 'Iron', 'H': 'Hydrogen', 'He': 'Helium', 'Hf': 'Hafnium',
    'Hg': 'Mercury', 'I': 'Iodine', 'In': 'Indium', 'Ir': 'Iridium', 'K': 'Potassium',
    'Kr': 'Krypton', 'La': 'Lanthanum', 'Li': 'Lithium', 'Mg': 'Magnesium', 'Mn': 'Manganese',
    'Mo': 'Molybdenum', 'N': 'Nitrogen', 'Na': 'Sodium', 'Nb': 'Niobium', 'Nd': 'Neodymium',
    'Ne': 'Neon', 'Ni': 'Nickel', 'O': 'Oxygen', 'Os': 'Osmium', 'P': 'Phosphorus',
    'Pr': 'Praseodymium', 'Pt': 'Platinum', 'Rb': 'Rubidium', 'Re': 'Rhenium', 'Rh': 'Rhodium',
    'S': 'Sulfur', 'Sc': 'Scandium', 'Si': 'Silicon', 'Sn': 'Tin', 'Sr': 'Strontium',
    'Ta': 'Tantalum', 'Ti': 'Titanium', 'Tm': 'Thulium', 'V': 'Vanadium', 'W': 'Tungsten',
    'Xe': 'Xenon', 'Y': 'Yttrium', 'Zr': 'Zirconium'
}

# Global metadata logger (for save_run_metadata) - Enhanced for full subset logging
metadata_log = {
    'run_timestamp': datetime.now(pytz.UTC).strftime("%Y%m%d_%H%M%S UTC"),
    'config_snapshot': {},  # Will fill with CONFIG copy
    'datasets_loaded': [],  # List of {'name': str, 'n_subsets': int, 'status': str}
    'subsets_processed': {},  # {dataset: {'n_subsets': int, 'D_f_values': list, 'hq_values': list, 'fluct_values': list, 'mean_D_f': float, ...}}
    'cca_results': {},  # {pair: {'correlations': list, 'mean_corr': float, 'D_f_X': float, 'D_f_Y': float}}
    'tci_pairs': {},  # {pair: {'corr': float, 'wavelet_corr': float, 'weight': float}}
    'mci_measurements': [],  # List of per-measurement corrs for full logging
    'tmci_folds': [],  # Cross-val folds for TMCI
    'errors': []  # List of error dicts
}

MAX_SUBSETS_IN_CSV = 5000        # How many rows in CSV.
SAVE_FULL_DETAILS_JSON = True    # Turn to True to save the JSON

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Run started: {timestamp}")

# Configuration for datasets
CONFIG = {
    'ligo_files': [
        '/content/drive/MyDrive/Datasets_UTMF/L-L1_GWOSC_O4a_16KHZ_R1-1368350720-4096.hdf5',
        '/content/drive/MyDrive/Datasets_UTMF/L-L1_GWOSC_04a_16KHZ_R1-1384779776-4096.hdf5',
        '/content/drive/MyDrive/Datasets_UTMF/L-L1_GWOSC_O4a_16KHZ_R1-1370202112-4096.hdf5',
        '/content/drive/MyDrive/Datasets_UTMF/L-L1_GWOSC_O4a_16KHZ_R1-1389420544-4096.hdf5'
    ],
    'ligo_names': ['LIGO-L1_1368350720', 'LIGO-L1_1384779776(GW231123)', 'LIGO-L1_1370202112', 'LIGO-L1_1389420544'],
    'ligo': [
        {'sample_rate': 16384,     # LIGO-L1_1368350720
         'total_duration': 4096,
         'subset_duration': 4,
         'n_subsets': 100,
         'freq_range': [1, 30],
         'expected_D_f': 1.22,
         'sigma_D_f': 0.05,
         'min_std': 1e-5,
         'scales': np.logspace(np.log10(8), np.log10(4 * 16384 / 16), 20, dtype=np.int32),
         'utmf_use': True
         },
        {'sample_rate': 16384,     # LIGO-L1_1384779776(GW231123)
         'total_duration': 4096,
         'subset_duration': 4,
         'n_subsets': 100,
         'freq_range': [1, 30],
         'expected_D_f': 1.22,
         'sigma_D_f': 0.05,
         'min_std': 1e-5,
         'scales': np.logspace(np.log10(8), np.log10(4 * 16384 / 16), 20, dtype=np.int32),
         'utmf_use': False
         },
        {'sample_rate': 16384,     # LIGO-L1_1370202112
         'total_duration': 4096,
         'subset_duration': 4,
         'n_subsets': 100,
         'freq_range': [1, 30],
         'expected_D_f': 1.22,
         'sigma_D_f': 0.05,
         'min_std': 1e-5,
         'scales': np.logspace(np.log10(8), np.log10(4 * 16384 / 16), 20, dtype=np.int32),
         'utmf_use':True
         },
        {'sample_rate': 16384,    # LIGO-L1_1389420544
         'total_duration': 4096,
         'subset_duration': 4,
         'n_subsets': 100,
         'freq_range': [1, 30],
         'expected_D_f': 1.22,
         'sigma_D_f': 0.05,
         'min_std': 1e-5,
         'scales': np.logspace(np.log10(8), np.log10(4 * 16384 / 16), 20, dtype=np.int32),
         'utmf_use': False
         }
    ],
    'cmb_files': [
        '/content/drive/MyDrive/Datasets_UTMF/COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits',
        '/content/drive/MyDrive/Datasets_UTMF/COM_CMB_IQU-smica_2048_R3.00_full.fits',
        '/content/drive/MyDrive/Datasets_UTMF/LFI_SkyMap_070_1024_R3.00_survey-1.fits'
    ],
    'cmb_names': ['Planck_CMB_I-Stokes_nosz', 'Planck_CMB_I-Stokes', 'Planck_LFI_70GHz'],
    'cmb': [
        {'nside': 2048,          # Planck_CMB_I-Stokes_nosz
         'subset_size': 100000,
         'n_subsets': 100,
         'expected_D_f': 1.19,
         'sigma_D_f': 0.04,
         'galactic_mask': True,
         'disc_radius': np.radians(20),
         'min_std': 1e-5,
         'fields': [0],
         'scales': np.array([2, 4, 8, 16, 32, 64, 128], dtype=np.int32),
         'utmf_use': False
         },
        {'nside': 2048,          # Planck_CMB_I-Stokes
         'subset_size': 100000,
         'n_subsets': 100,
         'expected_D_f': 1.19,
         'sigma_D_f': 0.04,
         'galactic_mask': True,
         'disc_radius': np.radians(20),
         'min_std': 1e-5,
         'fields': [0],
         'scales': np.array([2, 4, 8, 16, 32, 64, 128], dtype=np.int32),
         'utmf_use': True
         },
        {'nside': 1024,           # Planck_LFI_70GHz
         'subset_size': 45000,
         'n_subsets': 250,
         'expected_D_f': 1.19,
         'sigma_D_f': 0.04,
         'galactic_mask': True,
         'disc_radius': np.radians(30),
         'min_std': 1e-5,
         'fields': [0],
         'scales': np.array([2, 4, 8, 16, 32, 64, 128], dtype=np.int32),
         'utmf_use': False
         }
    ],
    'desi_file': '/content/drive/MyDrive/Datasets_UTMF/LRG_full.dat.fits',
    'desi_name': 'DESI_LRG',
    'desi': {'subset_size': 3700,
             'n_subsets': 100,
             'expected_D_f': 1.19,
             'sigma_D_f': 0.04,
             'min_std': 1e-5,
             'columns': ['FLUX_Z', 'FLUX_G', 'FLUX_R'],
             'scales': np.array([2, 4, 8, 16, 32, 64, 128], dtype=np.int32),
             'utmf_use': True
             },
    'cern_file': '/content/drive/MyDrive/Datasets_UTMF/data_B.exactly2lep.root',
    'cern_name': 'CERN_2Lepton',
    'cern': {'subset_size': 5000,
             'n_subsets': 100,
             'expected_D_f': 1.19,
             'sigma_D_f': 0.04,
             'min_std': 1e-5, 'tree': 'mini', 'columns': ['lep_pt', 'lep_eta', 'lep_phi'],
             'scales': np.array([2, 4, 8, 16, 32, 64, 128], dtype=np.int32),
             'utmf_use': True
             },
    'nist_file': '/content/drive/MyDrive/Datasets_UTMF/NIST_3.csv',
    'nist_name': 'NIST_3',  # Contains 63 elements
    'nist': {
        'elements': [],  # Initialize empty for loop
        'subset_size': lambda n: 50 if n >= 100 else max(10, n // 5),
        'n_subsets': 100,
        'expected_D_f': 1.16,
        'sigma_D_f': 0.04,
        'min_std': 1e-5,
        'columns': ['intens', 'wn(cm-1)', 'Aki(10^8 s^-1)', 'obs_wl_vac(nm)'],
        'elements_list': [  # All elements in NIST_3 file (63)
            ['Ac'], ['Ag'], ['Al'], ['Ar'], ['B'], ['Ba'], ['Be'], ['Bi'], ['Br'], ['C'],
            ['Ca'], ['Cd'], ['Ce'], ['Cl'], ['Co'], ['Cr'], ['Cs'], ['Cu'], ['Dy'], ['Eu'],
            ['F'], ['Fe'], ['H'], ['He'], ['Hf'], ['Hg'], ['I'], ['In'], ['Ir'], ['K'],
            ['Kr'], ['La'], ['Li'], ['Mg'], ['Mn'], ['Mo'], ['N'], ['Na'], ['Nb'], ['Nd'],
            ['Ne'], ['Ni'], ['O'], ['Os'], ['P'], ['Pr'], ['Pt'], ['Rb'], ['Re'], ['Rh'],
            ['S'], ['Sc'], ['Si'], ['Sn'], ['Sr'], ['Ta'], ['Ti'], ['Tm'], ['V'], ['W'],
            ['Xe'], ['Y'], ['Zr']
        ],
        'elements_list_utmf': [  # Element subsets for UTMF analysis
            ['Eu'], ['Fe'], ['Kr'], ['Sn'], ['Ni'], ['O']
        ],
        'scales': lambda n: np.logspace(np.log10(4), np.log10(min(n // 4, 500)), 15, dtype=np.int32),
        'utmf_use': True  # For elements in elements_list_utmf
    },
    'nanograv': {  # Contains 83 pulsars
        'base_dir': '/content/drive/MyDrive/Datasets_UTMF/NANOGrav15yr_PulsarTiming_v2.1.0',
        'file_templates': {
            'tim': 'narrowband/tim/{pulsar_name}_PINT_{date}.nb.tim',
            'par': 'narrowband/par/{pulsar_name}_PINT_{date}.nb.par',
            'res_full': 'residuals/{pulsar_name}_NG15yr_nb.full.res',
            'res_avg': 'residuals/{pulsar_name}_NG15yr_nb.avg.res',
            'dmx': 'narrowband/dmx/{pulsar_name}_dmxparse.nb.out',
            'noise': 'narrowband/noise/{pulsar_name}.nb.pars.txt',
            'template': 'narrowband/template/{pulsar_name}.sum.sm',
            'clock': 'clock/time_vla.dat'
        },
        'default_date': '20220302',
        'date_exceptions': {'J1713+0747': '20220309'},
        'pulsar_list': [  # Pulsars in NANOGrav file
            'J0030+0451', 'J0340+4130', 'J0406+3039', 'J0437-4715', 'J0509+0856',
            'J0557+1551', 'J0605+3757', 'J0610-2100', 'J0613-0200', 'J0614-3329',
            'J0636+5128', 'J0645+5158', 'J0709+0458', 'J0740+6620', 'J0931-1902',
            'J1012+5307', 'J1012-4235', 'J1022+1001', 'J1024-0719', 'J1125+7819',
            'J1312+0051', 'J1453+1902', 'J1455-3330', 'J1600-3053', 'J1614-2230',
            'J1630+3734', 'J1640+2224', 'J1643-1224', 'J1705-1903', 'J1713+0747',
            'J1719-1438', 'J1730-2304', 'J1738+0333', 'J1741+1351', 'J1744-1134',
            'J1745+1017', 'J1747-4036', 'J1802-2124', 'J1811-2405', 'J1832-0836',
            'J1843-1113', 'J1853+1303', 'J1903+0327', 'J1909-3744', 'J1910+1256',
            'J1911+1347', 'J1918-0642', 'J1923+2515', 'J1944+0907', 'J1946+3417',
            'J2010-1323', 'J2017+0603', 'J2022+2534', 'J2033+1734', 'J2043+1711',
            'J2124-3358', 'J2214+3000', 'J2229+2643', 'J2234+0611', 'J2234+0944',
            'J2302+4442', 'J2317+1439', 'J2322+2057', 'B1855+09', 'B1937+21',
            'B1953+29', 'J0751+1807', 'J0023+0923', 'J1751-2857', 'J0125-2327',
            'J0732+2314', 'J1221-0633', 'J1400-1431', 'J1630+3550', 'J2039-3616',
            'J0218+4232', 'J0337+1715', 'J0621+2514', 'J0721-2038', 'J1803+1358',
            'B1257+12', 'J1327+3423', 'J0154+1833'
        ],
        'pulsar_list_utmf': [  # Pulsars for UTMF analysis
            'J1713+0747', 'J1730-2304', 'J1738+0333', 'J1741+1351', 'J2017+0603',
            'J0709+0458'
        ],
         'subset_size': 250,
         'n_subsets': 100,
         'expected_D_f': 1.19,
         'sigma_D_f': 0.04,
         'min_std': 1e-5,
         'columns': ['residuals', 'dmx', 'red_noise'],
         'scales': np.array([2, 4, 8, 16, 32, 64, 128, 256], dtype=np.int32),
         'utmf_use': True  # For pulsars in pulsar_list_utmf
    },
    'qrng': {'subset_size': 2560,
             'n_subsets': 100,
             'expected_D_f': 1.21,
             'sigma_D_f': 0.04,
             'min_std': 1e-5,
             'scales': np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], dtype=np.int32),
             'utmf_use': True
             },
    'gaia': {'file': '/content/drive/MyDrive/Datasets_UTMF/gaia_dr3.tsv',
             'name': 'Gaia_DR3',
             'subset_size': 5000,
             'n_subsets': 100,
             'expected_D_f': 1.19,
             'sigma_D_f': 0.04,
             'min_std': 1e-5,
             'columns': ['RA_ICRS', 'DE_ICRS', 'pmRA', 'pmDE', 'Gmag'],
             'sep': '\t',
             'scales': np.array([1, 2, 4, 6, 8, 16, 32, 64], dtype=np.int32),
             'utmf_use': True
             },
    'mfdfa': {
              'q_values': np.arange(-8, 8.2, 0.2), 
              'detrend_order': 0
              },           
    # UTMF thresholds
    'utmf': {
             'd_f_min_threshold': 1.0, 
             'd_f_max_threshold': 2.0
             },
    # Nieuw: Flags voor computations
    'compute_indices': True,  # Turn TCI/MCI/TMCI on/off
    'cross_val': True,       # Simple k=3 fold for TMCI std (voor p<0.05)
    # Metadata
    'metadata': {'save_flag': True, 'log_level': 'info'}
}

# Numba-compatible linear detrending
@jit(nopython=True)
def polyfit_linear(x, y, lambda_reg=1e-5):
    # Linear polyfit with regularization for stability
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)
    denom = n * sum_x2 - sum_x**2 + lambda_reg
    if abs(denom) < 1e-10:
        return np.array([0.0, sum_y / n])
    m = (n * sum_xy - sum_x * sum_y) / denom
    b = (sum_y * sum_x2 - sum_x * sum_xy) / denom
    return np.array([m, b])

@jit(nopython=True)
def polyval_linear(coeffs, x):
    return coeffs[0] * x + coeffs[1]

# Unified MFDFA function
@jit(nopython=True)
def jedi_mfdfa(data, scales, q_values, detrend_order=0):
    n = len(data)
    fluct = np.zeros((len(q_values), len(scales)))
    rms_values = []
    slopes = np.zeros(len(q_values))

    for i in range(len(scales)):
        s = scales[i]
        segments = n // s
        if segments < 2:
            fluct[:, i] = np.nan
            continue
        rms = np.zeros(segments)
        valid_segments = 0
        for v in range(segments):
            segment = data[v*s:(v+1)*s]
            if len(segment) != s or np.std(segment) < 1e-10:
                continue
            x = np.arange(s, dtype=np.float64)
            if detrend_order > 0:
                try:
                    coeffs = polyfit_linear(x, segment)
                    trend = polyval_linear(coeffs, x)
                    detrended = segment - trend
                except:
                    detrended = segment - np.sum(segment) / s
            else:
                detrended = segment - np.sum(segment) / s
            sum_squares = 0.0
            for j in range(s):
                sum_squares += detrended[j]**2
            rms_val = np.sqrt(sum_squares / s + 1e-12)
            if rms_val > 1e-10:
                rms[valid_segments] = rms_val
                valid_segments += 1
        if valid_segments < 2:
            fluct[:, i] = np.nan
            continue
        rms = rms[:valid_segments]
        rms_values.append(rms)
        for j in range(len(q_values)):
            q = q_values[j]
            if q == 0:
                sum_log = 0.0
                count = 0
                for k in range(valid_segments):
                    if rms[k] > 1e-10:
                        sum_log += np.log(rms[k]**2 + 1e-12)
                        count += 1
                fluct[j, i] = np.exp(0.5 * (sum_log / count)) if count > 0 else np.nan
            else:
                sum_power = 0.0
                count = 0
                for k in range(valid_segments):
                    if rms[k] > 1e-10:
                        sum_power += (rms[k] + 1e-12)**q
                        count += 1
                fluct[j, i] = (sum_power / count)**(1/q) if count > 0 else np.nan
                if not np.isfinite(fluct[j, i]) or fluct[j, i] <= 0:
                    fluct[j, i] = np.nan
    valid_scales = np.sum(np.isfinite(fluct), axis=0)
    if np.max(valid_scales) < 4:
        return np.nan, np.full(len(q_values), np.nan), rms_values, fluct, slopes
    for j in range(len(q_values)):
        valid = np.isfinite(fluct[j, :]) & (fluct[j, :] > 0)
        if np.sum(valid) < 4:
            slopes[j] = np.nan
            continue
        coeffs = np.zeros(2)
        X = np.log(scales[valid])
        Y = np.log(fluct[j, valid] + 1e-12)
        n_valid = len(X)
        sum_x = np.sum(X)
        sum_y = np.sum(Y)
        sum_xy = np.sum(X * Y)
        sum_x2 = np.sum(X * X)
        denom = n_valid * sum_x2 - sum_x**2 + 1e-5
        if abs(denom) > 1e-10:
            coeffs[0] = (n_valid * sum_xy - sum_x * sum_y) / denom
            coeffs[1] = (sum_y * sum_x2 - sum_x * sum_xy) / denom
        slopes[j] = coeffs[0] if coeffs[0] > 0 else np.nan
    hq = slopes
    valid_hq = np.isfinite(hq)
    if np.sum(valid_hq) >= 2:
        tau = hq * q_values - 1
        alpha = np.diff(tau[valid_hq]) / np.diff(q_values[valid_hq])
        f_alpha = q_values[valid_hq][1:] * alpha - tau[valid_hq][1:]
        D_f = np.nanmean(alpha) if np.isfinite(alpha).any() else np.nan
    else:
        D_f = np.nan
    return D_f, hq, rms_values, fluct, slopes

# Denoising function
def denoise_data(data, data_type, ligo_idx=None, cmb_idx=None):
    try:
        std_before = np.std(data)
       # print(f"Standard deviation before denoising ({data_type}): {std_before:.2e}") # Remove first # to see denoising
        if data_type == 'ligo':
            denoised = data * 1e16
            denoised = scipy.signal.savgol_filter(denoised, window_length=7, polyorder=1, mode='nearest')
        elif data_type == 'nist':
            data = np.log1p(np.abs(data))
            coeffs = pywt.wavedec(data, 'db4', level=4 if data_type == 'nist' else 5)
            threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(data)))
            coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
            denoised = pywt.waverec(coeffs, 'db4')
            denoised = denoised[:len(data)]
            denoised = scipy.signal.savgol_filter(denoised, window_length=5, polyorder=1, mode='nearest')
        elif data_type == 'nanograv':
            coeffs = pywt.wavedec(data, 'db4', level=5)
            threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(data)))
            coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
            denoised = pywt.waverec(coeffs, 'db4')
            denoised = denoised[:len(data)]
            denoised = scipy.signal.savgol_filter(denoised, window_length=11, polyorder=2, mode='nearest')
        elif data_type == 'qrng':
            # Normalize randomness (no trend, but std=1 for coherence)
            denoised = (data - np.mean(data)) / (np.std(data) + 1e-10)
        else:  # cmb, desi, cern, gaia
            denoised = data / np.std(data)
        std_after = np.std(denoised)
        min_std = CONFIG[data_type]['min_std'] if data_type not in ['ligo', 'cmb'] else \
                  CONFIG['ligo'][ligo_idx]['min_std'] if data_type == 'ligo' else \
                  CONFIG['cmb'][cmb_idx]['min_std']
       # print(f"Standard deviation after denoising ({data_type}): {std_after:.2e}") # Remove first # to see denoising
        if std_after < min_std:
            print(f"Warning: Low variability after denoising ({data_type}, std={std_after:.2e})")
        return denoised.astype(np.float64)
    except Exception as e:
        print(f"Error in denoising {data_type}: {e}")
        return data.astype(np.float64)


# Extraction tool for TCI (time-series) and MCI (measurements)
def extract_tci_mci_data(raw_data, data_type, ligo_idx=None, cmb_idx=None):
    MAX_LEN = 1000000  # Downsample threshold
    if len(raw_data) > MAX_LEN:
        indices = np.linspace(0, len(raw_data)-1, MAX_LEN, dtype=int)
        raw_data = raw_data[indices]
        print(f"Downsampled {data_type} data from ~{len(raw_data) * (len(raw_data) / MAX_LEN):.0f} to {MAX_LEN} samples for RAM.")
    if data_type == 'ligo':
        tci_extracted = denoise_data(raw_data, 'ligo', ligo_idx=ligo_idx) 
        _, psd = scipy.signal.welch(tci_extracted, fs=CONFIG['ligo'][ligo_idx]['sample_rate'], nperseg=min(256, len(tci_extracted)//4))
        psd_norm = psd / (np.max(psd) + 1e-10)
        psd_padded = np.tile(psd_norm, (len(tci_extracted) // len(psd_norm) + 1, 1))[:len(tci_extracted), :min(8, len(psd_norm))]
        mci_extracted = psd_padded
    elif data_type == 'cmb':
        tci_extracted = denoise_data(raw_data, 'cmb', cmb_idx=cmb_idx)
        # MCI via PSD 
        f, psd = scipy.signal.welch(tci_extracted, fs=1.0, nperseg=min(1024, len(tci_extracted)//8))
        psd_norm = psd / (np.max(psd) + 1e-10)
        mci_extracted = np.tile(psd_norm, (len(tci_extracted)//len(psd_norm)+1, 1))[:len(tci_extracted), :8]
    elif data_type in ['desi', 'gaia', 'cern', 'nist', 'nanograv']:
        # TCI = 1st kolom (or residue), MCI = the rest → without wavelet
        if isinstance(raw_data, list):
            signals = raw_data
        else:
            signals = [raw_data[:, k] if raw_data.ndim > 1 else raw_data for k in range(raw_data.shape[1] if raw_data.ndim > 1 else 1)]

        tci_extracted = denoise_data(signals[0], data_type)
        mci_extracted = np.column_stack([denoise_data(s, data_type) for s in signals[1:]]) if len(signals)>1 else tci_extracted.reshape(-1,1)

    elif data_type == 'qrng':
        tci_extracted = denoise_data(raw_data, 'qrng')  
        _, psd = scipy.signal.welch(tci_extracted, fs=1.0, nperseg=min(256, len(tci_extracted)//4))
        psd_norm = psd / (np.max(psd) + 1e-10)
        psd_padded = np.tile(psd_norm, (len(tci_extracted) // len(psd_norm) + 1, 1))[:len(tci_extracted), :min(8, len(psd_norm))]
        mci_extracted = psd_padded
    else:
        raise ValueError(f"Invalid data_type for extraction: {data_type}")

    # Consistent length
    min_len = min(len(tci_extracted), mci_extracted.shape[0] if mci_extracted.ndim > 1 else len(mci_extracted))
    tci_extracted = tci_extracted[:min_len]
    if mci_extracted.ndim > 1:
        mci_extracted = mci_extracted[:min_len, :]
    else:
        mci_extracted = mci_extracted[:min_len]

    gc.collect()
    return tci_extracted, mci_extracted

# Dynamic date selection for NANOGrav
def get_pulsar_date(pulsar_name, base_dir, template):
    files = glob.glob(os.path.join(base_dir, template.format(pulsar_name=pulsar_name, date='*')))
    if files:
        try:
            return files[0].split('_PINT_')[1].split('.nb.tim')[0]
        except IndexError:
            print(f"Warning: Cannot extract date for {pulsar_name}, using default")
    return CONFIG['nanograv']['default_date']

# Data loading (full rebuild: snippet-based gaia, 5 cols, aligned dropna, scalar clip)
def load_data(file_path, data_type, ligo_idx=None, cmb_idx=None, pulsar_name=None):
    try:
        if data_type == 'ligo':
            print(f"Loading LIGO file: {file_path}")
            with h5py.File(file_path, 'r') as f:
                if 'strain' not in f or 'Strain' not in f['strain']:
                    raise KeyError("Key 'strain/Strain' not found")
                strain = f['strain']['Strain'][:]
                sos = scipy.signal.butter(2, CONFIG['ligo'][ligo_idx]['freq_range'], btype='band',
                                         fs=CONFIG['ligo'][ligo_idx]['sample_rate'], output='sos')
                strain_filtered = scipy.signal.sosfilt(sos, strain)
                data = da.from_array(strain_filtered, chunks='auto').compute().astype(np.float64)
                for_mfdfa = data
                tci_extracted, mci_extracted = extract_tci_mci_data(data, data_type, ligo_idx=ligo_idx)
                return for_mfdfa, tci_extracted, mci_extracted
        elif data_type == 'cmb':
            print(f"Loading CMB file: {file_path}")
            cmb_maps = hp.read_map(file_path, field=CONFIG['cmb'][cmb_idx]['fields'], verbose=False)
            if isinstance(cmb_maps, np.ndarray):
                cmb_maps = [cmb_maps]
            if CONFIG['cmb'][cmb_idx]['galactic_mask']:
                npix = hp.nside2npix(CONFIG['cmb'][cmb_idx]['nside'])
                mask = np.ones(npix, dtype=bool)
                galactic_pixels = hp.query_strip(CONFIG['cmb'][cmb_idx]['nside'], np.radians(60), np.radians(120))
                mask[galactic_pixels] = False
                cmb_maps = [cmb_map[mask] for cmb_map in cmb_maps]
            data = da.from_array(cmb_maps[0], chunks='auto').compute().astype(np.float64)
            for_mfdfa = data
            tci_extracted, mci_extracted = extract_tci_mci_data(data, data_type, cmb_idx=cmb_idx)
            return for_mfdfa, tci_extracted, mci_extracted
        elif data_type == 'desi':
            print(f"Loading DESI file: {file_path}")
            with fits.open(file_path) as hdul:
                signals = []
                for column in CONFIG['desi']['columns']:
                    data_col = hdul[1].data[column].astype(np.float64)
                    data_col = np.nan_to_num(data_col, nan=np.nanmedian(data_col), posinf=np.nanmedian(data_col), neginf=np.nanmedian(data_col))
                    data_col = data_col[np.isfinite(data_col)]
                    if len(data_col) < 10:
                        print(f"Too few data for DESI column {column}: {len(data_col)}")
                        continue
                    signals.append(data_col)
                if not signals:
                    print("No valid data for DESI")
                    return None, None, None
                for_mfdfa = signals[0]
                min_length = min(len(s) for s in signals)
                signals = [s[:min_length] for s in signals]
                tci_extracted, mci_extracted = extract_tci_mci_data(signals, data_type)
                return for_mfdfa, tci_extracted, mci_extracted
        elif data_type == 'gaia':
            print(f"Loading Gaia DR3 TSV: {file_path}")
            try:
                skip_rows = 257  # Skip to data (post-header/units/dashes, from snippet)
                data = pd.read_csv(
                    file_path,
                    sep='\t',
                    skiprows=skip_rows,
                    nrows=CONFIG['gaia']['subset_size'] * 2,  # *2 for sample, then drop
                    header=None,  # No header
                    engine='python',
                    on_bad_lines='skip',
                    quoting=3,
                    comment=None
                )
                print(f"Loaded Gaia DR3: {len(data)} rows, {len(data.columns)} cols")
                col_map = {
                    'RA_ICRS': 1,
                    'DE_ICRS': 2,
                    'pmRA': 12,  # From snippet
                    'pmDE': 14,
                    'Gmag': 54
                }
                available_cols = []
                signals = []
                for col_name, idx in col_map.items():
                    if idx < len(data.columns):
                        col_data = pd.to_numeric(data.iloc[:, idx], errors='coerce')
                        signals.append(col_data.dropna().values.astype(np.float64))
                        available_cols.append(col_name)
                        print(f"Loaded {col_name}: {len(signals[-1])} values, mean={np.mean(signals[-1]):.3f}")
                    else:
                        print(f"Warning: Col {col_name} (idx {idx}) beyond {len(data.columns)} cols; skip")
                if not signals:
                    raise ValueError("No signals loaded – check snippet indices")
                print(f"Available cols: {available_cols}")
                min_length = min(len(s) for s in signals)
                signals = [s[:min_length] for s in signals]
                data_array = np.column_stack(signals)
                for i, col in enumerate(available_cols):
                    col_data = data_array[:, i]
                    q_low = np.quantile(col_data, 0.01)
                    q_high = np.quantile(col_data, 0.99)
                    data_array[:, i] = np.clip(col_data, q_low, q_high)
                    print(f"Clipped {col}: [{q_low:.3f}, {q_high:.3f}]")
                if 'RA_ICRS' in available_cols and 'DE_ICRS' in available_cols:
                    ra_idx = available_cols.index('RA_ICRS')
                    de_idx = available_cols.index('DE_ICRS')
                    data_array[:, ra_idx] = np.deg2rad(data_array[:, ra_idx])
                    data_array[:, de_idx] = np.deg2rad(data_array[:, de_idx])
                    print("RA/DE converted to radians")
                if min_length < 10:
                    print("Too few valid data for Gaia")
                    return None, None, None
                for_mfdfa = np.mean(data_array, axis=1)
                tci_extracted, mci_extracted = extract_tci_mci_data(data_array, data_type)

                print(f"Gaia loaded: {len(signals)} cols, length {min_length}")
                return for_mfdfa, tci_extracted, mci_extracted

            except Exception as e:
                print(f"Error loading Gaia: {e}")
                import traceback
                traceback.print_exc()
                return None, None, None
        elif data_type == 'cern':
            print(f"Loading CERN file: {file_path}")
            with uproot.open(file_path) as f:
                signals = []
                for column in CONFIG['cern']['columns']:
                    data = f[CONFIG['cern']['tree']][column].array(library='np')
                    if isinstance(data, np.ndarray):
                        if column == 'lep_pt':
                            data = np.concatenate([np.array(x).flatten() for x in data if len(x) > 0])
                        else:
                            data = np.array([np.mean(x) if len(x) > 0 else np.nan for x in data])
                    data = np.nan_to_num(data, nan=np.nanmedian(data), posinf=np.nanmedian(data), neginf=np.nanmedian(data))
                    data = data[np.isfinite(data)]
                    if len(data) < 10:
                        print(f"Too few data for CERN column {column}: {len(data)}")
                        continue
                    signals.append(data)
                if not signals:
                    print("No valid data for CERN")
                    return None, None, None
                for_mfdfa = signals[0]
                min_length = min(len(s) for s in signals)
                signals = [s[:min_length] for s in signals]
                tci_extracted, mci_extracted = extract_tci_mci_data(signals, data_type)
                return for_mfdfa, tci_extracted, mci_extracted
        elif data_type == 'nist':
            print(f"Loading NIST file: {file_path}")
            dtypes = {
                'sp_num': str, 'Aki(10^8 s^-1)': str, 'wn(cm-1)': str, 'intens': str,
                'tp_ref': str, 'line_ref': str
            }
            data = pd.read_csv(file_path, dtype=dtypes, na_values=['', 'nan', 'NaN', '"'])
            elements = CONFIG['nist']['elements']
            elements_str = '_'.join(elements)  # e.g., 'Ac'
            full_element = ELEMENT_NAMES.get(elements[0], elements[0])  # e.g., 'Actinium'
            print(f"[NIST] Loading for element: {elements_str} ({full_element})")
            data = data[data['element'].isin(elements)]
            signals = []
            for column in CONFIG['nist']['columns']:
                data[column] = data[column].str.strip('="').replace('', np.nan)
                data[column] = pd.to_numeric(data[column], errors='coerce')
                q_low, q_high = data[column].quantile([0.025, 0.975])
                signal = data[(data[column] >= q_low) & (data[column] <= q_high)][column].dropna().values
                if len(signal) < 10:
                    print(f"Too few data for {full_element} ({elements_str}), column {column}: {len(signal)}")
                    continue
                signals.append(signal)
            if not signals:
                print(f"No valid data for {full_element} ({elements_str})")
                return None, None, None, None  # 4-tuple for consistency
            for_mfdfa = signals[0]
            min_length = min(len(s) for s in signals)
            signals = [s[:min_length] for s in signals]
            tci_extracted, mci_extracted = extract_tci_mci_data(signals, data_type)
            return for_mfdfa, tci_extracted, mci_extracted, full_element
        elif data_type == 'nanograv':
            print(f"Loading NANOGrav data for pulsar: {pulsar_name}")
            base_dir = CONFIG['nanograv']['base_dir']
            date = get_pulsar_date(pulsar_name, base_dir, CONFIG['nanograv']['file_templates']['tim'])
            files = {
                'tim': os.path.join(base_dir, CONFIG['nanograv']['file_templates']['tim'].format(pulsar_name=pulsar_name, date=date)),
                'par': os.path.join(base_dir, CONFIG['nanograv']['file_templates']['par'].format(pulsar_name=pulsar_name, date=date)),
                'res_full': os.path.join(base_dir, CONFIG['nanograv']['file_templates']['res_full'].format(pulsar_name=pulsar_name)),
                'res_avg': os.path.join(base_dir, CONFIG['nanograv']['file_templates']['res_avg'].format(pulsar_name=pulsar_name)),
                'dmx': os.path.join(base_dir, CONFIG['nanograv']['file_templates']['dmx'].format(pulsar_name=pulsar_name)),
                'noise': os.path.join(base_dir, CONFIG['nanograv']['file_templates']['noise'].format(pulsar_name=pulsar_name)),
                'clock': os.path.join(base_dir, CONFIG['nanograv']['file_templates']['clock'])
            }
            residuals = None
            for res_file, res_type in [(files['res_full'], 'full'), (files['res_avg'], 'avg')]:
                if os.path.exists(res_file):
                    try:
                        df = pd.read_csv(res_file, delim_whitespace=True, comment='#', header=None)
                        res_col = 1
                        if df.shape[1] <= res_col:
                            print(f"Error: No residual column ({res_col}) in {res_file}")
                            continue
                        residuals = pd.to_numeric(df.iloc[:, res_col], errors='coerce').dropna().values
                        print(f"Loaded {len(residuals)} residuals from {res_file}")
                        break
                    except Exception as e:
                        print(f"Error loading {res_type} residuals: {e}")
            if residuals is None:
                if not os.path.exists(files['tim']) or not os.path.exists(files['par']):
                    print(f"Error: .tim or .par file not found for {pulsar_name}")
                    return None, None, None
                try:
                    model, toas = get_model_and_toas(files['par'], files['tim'], planets=True)
                    print(f"Model components for {pulsar_name}: {list(model.components.keys())}")
                    model.validate()
                    if os.path.exists(files['clock']):
                        with open(files['clock'], 'r') as f:
                            clock_data = pd.read_csv(f, delim_whitespace=True, comment='#', header=None)
                            clock_corrections = clock_data.iloc[:, 1].values
                            toas.adjust_TOAs(clock_corrections * 1e6)
                    residuals = model.residuals(toas, use_weighted_mean=False).to('s').value
                    residuals = residuals[np.isfinite(residuals)]
                    print(f"Loaded residuals: {len(residuals)} points")
                except Exception as e:
                    print(f"Error loading with pint: {e}")
                    return None, None, None
            if len(residuals) < CONFIG['nanograv']['subset_size']:
                print(f"Error: Insufficient residuals: {len(residuals)} < {CONFIG['nanograv']['subset_size']}")
                return None, None, None
            dmx_data = None
            if os.path.exists(files['dmx']):
                try:
                    df_dmx = pd.read_csv(files['dmx'], delim_whitespace=True, comment='#', header=None)
                    dmx_data = pd.to_numeric(df_dmx.iloc[:, 1], errors='coerce').dropna().values
                    print(f"Loaded DMX data: {len(dmx_data)} points")
                except Exception as e:
                    print(f"Error loading DMX data: {e}")
            red_noise = None
            if os.path.exists(files['noise']):
                try:
                    with open(files['noise'], 'r') as f:
                        noise_lines = f.readlines()
                    for line in noise_lines:
                        if 'EFAC' in line or 'EQUAD' in line or 'ECORR' in line:
                            red_noise = float(line.split()[1]) if red_noise is None else red_noise
                    print(f"Loaded red noise parameter: {red_noise}")
                except Exception as e:
                    print(f"Error loading noise parameters: {e}")
            data_dict = {'residuals': residuals}
            if dmx_data is not None:
                data_dict['dmx'] = dmx_data
            if red_noise is not None:
                data_dict['red_noise'] = np.full_like(residuals, red_noise)
            min_length = min(len(data) for data in data_dict.values())
            data_array = np.column_stack([data[:min_length] for data in data_dict.values()])
            for_mfdfa = residuals
            tci_extracted, mci_extracted = extract_tci_mci_data(data_array, data_type)
            return for_mfdfa, tci_extracted, mci_extracted
        elif data_type == 'qrng':
            print("Loading LFDR QRNG via API...")
            try:
                data_list = []
                for i in range(50):  # 50 calls for ~100k bits
                    for retry in range(3):
                        url = 'https://lfdr.de/qrng_api/qrng?length=256&format=HEX'
                        response = requests.get(url, timeout=5)
                        if response.status_code == 200:
                            try:
                                json_data = response.json()
                                hex_string = json_data['qrn']
                                bits = []
                                for char in hex_string:
                                    byte = int(char, 16)
                                    bits.extend([int(b) for b in f"{byte:04b}"])
                                data_list.extend(bits)
                                print(f"QRNG call {i+1}: {len(bits)} bits geladen")
                                break
                            except (json.JSONDecodeError, KeyError) as je:
                                print(f"JSON/Key error, retry {retry+1}: {je}. Response: {response.text[:100]}...")
                                if retry == 2:
                                    raise ValueError("API parse failed")
                        else:
                            print(f"HTTP {response.status_code}, retry {retry+1}. Response: {response.text[:100]}...")
                            if retry < 2:
                                time.sleep(1)
                            else:
                                raise ValueError("API failed after retries")
                if len(data_list) < 100:
                    raise ValueError(f"Not enough QRNG-data: {len(data_list)} bits")
                data = np.array(data_list).astype(np.float64)
                data += np.random.normal(0, 1e-6, len(data))
                print(f"Geladen {len(data)} quantum random bits")
                for_mfdfa = data
                tci_extracted, mci_extracted = extract_tci_mci_data(data, data_type)
                return for_mfdfa, tci_extracted, mci_extracted
            except Exception as e:
                print(f"Error loading QRNG (fallback to simulation): {e}")
                data = np.random.randint(0, 2, 102400).astype(np.float64)
                data += np.random.normal(0, 1e-6, len(data))
                print(f"Fallback: Simulated {len(data)} random samples")
                for_mfdfa = data
                tci_extracted, mci_extracted = extract_tci_mci_data(data, data_type)
                return for_mfdfa, tci_extracted, mci_extracted
        else:
            raise ValueError("Invalid data_type")
    except Exception as e:
        print(f"Error loading {data_type}: {e}")
        return None, None, None

# Z-test validation for MFDFA
def validate_df(D_f_values, expected_D_f, sigma_D_f):
    valid_D_f = [x for x in D_f_values if np.isfinite(x)]
    if not valid_D_f:
        print("No valid D_f values for validation")
        return np.nan
    mean_D_f = np.nanmean(valid_D_f)
    std_D_f = np.nanstd(valid_D_f)
    n = len(valid_D_f)
    if std_D_f < 1e-10 or n < 2:
        print("Insufficient variability or data for Z-test")
        return np.nan
    z_score = (mean_D_f - expected_D_f) / (std_D_f / np.sqrt(n))
    p_value = 2 * (1 - norm.cdf(abs(z_score)))
    return p_value

# Check reliability of h_q
def is_valid_hq(hq, q_values, min_valid_ratio=0.6, monotonicity_tolerance=0.05, require_monotonicity=True):
    if isinstance(hq, np.floating):
        return False
    valid_hq = np.isfinite(hq)
    valid_ratio = np.sum(valid_hq) / len(hq)
    if valid_ratio < min_valid_ratio:
        return False
    if not require_monotonicity:
        return True
    valid_indices = np.where(valid_hq)[0]
    if len(valid_indices) < 2:
        return False
    hq_valid = hq[valid_indices]
    q_valid = q_values[valid_indices]
    diff_hq = np.diff(hq_valid) / np.diff(q_valid)
    monotonic = np.all(diff_hq <= monotonicity_tolerance)
    return monotonic

# Subset processing - Enhanced logging for every subset
def process_subset(subset_idx, data, data_type, dataset_name, scales, ligo_idx=None, cmb_idx=None):
    """
    Processes a single subset: Extract → Denoise → MFDFA.
    Returns: D_f, hq, fluct, slopes, subset_data_denoised (no logging here - done in process_dataset)
    """
    try:
        # Subset extraction (same as before)
        if data_type == 'ligo':
            n_samples = int(CONFIG['ligo'][ligo_idx]['subset_duration'] * CONFIG['ligo'][ligo_idx]['sample_rate'])
            max_start = len(data) - n_samples
            subset_start = np.random.randint(0, max_start + 1)
            subset_data = data[subset_start:subset_start + n_samples]
            if len(subset_data) != n_samples:
                subset_data = np.pad(subset_data, (0, n_samples - len(subset_data)), mode='constant')
        elif data_type == 'cmb':
            nside = CONFIG['cmb'][cmb_idx]['nside']
            subset_size = CONFIG['cmb'][cmb_idx]['subset_size']
            npix = len(data)
            valid_indices = np.arange(npix)
            center_pix = np.random.choice(valid_indices)
            try:
                subset_indices = hp.query_disc(nside, hp.pix2vec(nside, center_pix, nest=False), radius=CONFIG['cmb'][cmb_idx]['disc_radius'])
                valid_mask = subset_indices < npix
                subset_indices = subset_indices[valid_mask]
                if len(subset_indices) < subset_size:
                    subset_indices = np.random.choice(valid_indices, size=subset_size, replace=False)
                elif len(subset_indices) > subset_size:
                    subset_indices = np.random.choice(subset_indices, size=subset_size, replace=False)
                subset_data = data[subset_indices]
                if np.std(subset_data) < CONFIG['cmb'][cmb_idx]['min_std']:
                    subset_indices = np.random.choice(valid_indices, size=subset_size, replace=False)
                    subset_data = data[subset_indices]
            except Exception:
                subset_indices = np.random.choice(valid_indices, size=subset_size, replace=False)
                subset_data = data[subset_indices]
        elif data_type in ['desi', 'cern', 'nist', 'nanograv', 'qrng']:
            subset_size = CONFIG[data_type]['subset_size'] if 'subset_size' in CONFIG[data_type] else len(data) // 4
            if data_type == 'nist':
                subset_size = CONFIG[data_type]['subset_size'](len(data))
            max_start = len(data) - subset_size
            if max_start < 0:
                print(f"Error: Data too short for {dataset_name} (length={len(data)}, required={subset_size})")
                q_len = len(CONFIG['mfdfa']['q_values'])
                return np.nan, np.full(q_len, np.nan), np.full((q_len, len(scales)), np.nan), np.full(q_len, np.nan), None
            subset_start = np.random.randint(0, max_start + 1)
            subset_data = data[subset_start:subset_start + subset_size]
            if len(subset_data) != subset_size:
                subset_data = np.pad(subset_data, (0, subset_size - len(subset_data)), mode='constant')
        elif data_type == 'gaia':
            subset_size = CONFIG['gaia']['subset_size']
            max_start = len(data) - subset_size
            if max_start < 0:
                q_len = len(CONFIG['mfdfa']['q_values'])
                return np.nan, np.full(q_len, np.nan), np.full((q_len, len(scales)), np.nan), np.full(q_len, np.nan), None
            subset_start = np.random.randint(0, max_start + 1)
            subset_data = data[subset_start:subset_start + subset_size]
            if len(subset_data) != subset_size:
                subset_data = np.pad(subset_data, (0, subset_size - len(subset_data)), mode='constant', constant_values=0)
        else:
            raise ValueError("Invalid data_type")

        # Denoise
        subset_data_denoised = denoise_data(subset_data, data_type, ligo_idx, cmb_idx)

        # MFDFA input (for gaia: pairwise dists)
        if data_type == 'gaia' and subset_data_denoised.ndim > 1:
            ra = subset_data_denoised[:, 0]  # RA
            de = subset_data_denoised[:, 1]  # DE
            pmra = subset_data_denoised[:, 2]  # pmRA
            pmde = subset_data_denoised[:, 3]  # pmDE
            sample_idx = np.random.choice(len(ra), 50, replace=False)
            ra_s, de_s, pmra_s, pmde_s = ra[sample_idx], de[sample_idx], pmra[sample_idx], pmde[sample_idx]
            dists = []
            for i in range(len(ra_s)):
                for j in range(i+1, len(ra_s)):
                    d_pos = np.sqrt((ra_s[i] - ra_s[j])**2 + (de_s[i] - de_s[j])**2)
                    d_vel = np.sqrt((pmra_s[i] - pmra_s[j])**2 + (pmde_s[i] - pmde_s[j])**2)
                    d = np.sqrt(d_pos**2 + d_vel**2)  # 3D (pos + vel)
                    dists.append(d)
            dists = np.array(dists)
            if len(dists) < 100:
                dists = np.pad(dists, (0, 100 - len(dists)), mode='constant')
            mfdfa_input = np.log(dists + 1)
        else:
            mfdfa_input = subset_data_denoised

        # MFDFA
        D_f, hq, rms_values, fluct, slopes = jedi_mfdfa(mfdfa_input, scales, CONFIG['mfdfa']['q_values'], CONFIG['mfdfa']['detrend_order'])

        return D_f, hq, fluct, slopes, subset_data_denoised
    except Exception as e:
        metadata_log['errors'].append({'subset': subset_idx, 'dataset': dataset_name, 'error': str(e)})  # Dit is main-safe, want append na return
        print(f"Error in subset {subset_idx+1} of {dataset_name}: {e}")
        q_len = len(CONFIG['mfdfa']['q_values'])
        return np.nan, np.full(q_len, np.nan), np.full((q_len, len(scales)), np.nan), np.full(q_len, np.nan), None
    finally:
        gc.collect()

# process_dataset - Enhanced to collect full lists
def process_dataset(data, data_type, dataset_name, scales, expected_D_f, sigma_D_f, ligo_idx=None, cmb_idx=None):
    if data is None:
        print(f"No valid data for {dataset_name}. Skipping.")
        metadata_log['errors'].append({'dataset': dataset_name, 'error': 'No data'})
        return None, []
    n_subsets = CONFIG[data_type]['n_subsets'] if data_type not in ['ligo', 'cmb'] else \
                CONFIG['ligo'][ligo_idx]['n_subsets'] if data_type == 'ligo' else \
                CONFIG['cmb'][cmb_idx]['n_subsets']
    if data_type == 'gaia':
        n_subsets = CONFIG['gaia']['n_subsets']
    results = Parallel(n_jobs=-1)(
        delayed(process_subset)(subset_idx, data, data_type, dataset_name, scales, ligo_idx=ligo_idx, cmb_idx=cmb_idx)
        for subset_idx in tqdm(range(n_subsets), desc=f"Subsets for {dataset_name}")
    )
    D_f_values = [r[0] for r in results]
    hq_values = [r[1] for r in results if r[1] is not None]  # full h(q)-curves per subset
    hq_means = [np.nanmean(hq) for hq in hq_values]         
    fluct_values = [r[2] for r in results]
    slopes_values = [r[3] for r in results]
    subset_data_list = [r[4] for r in results]
    print(f"\nSummary for {dataset_name}:")
    if D_f_values and np.any(np.isfinite(D_f_values)):
        mean_D_f = np.nanmean(D_f_values)
        std_D_f = np.nanstd(D_f_values)
        mean_hq = np.nanmean([np.nanmean(hq) for hq in hq_values if np.any(np.isfinite(hq))])
        print(f"Mean D_f: {mean_D_f:.3f} ± {std_D_f:.3f}")
        print(f"Expected D_f: {expected_D_f} ± {sigma_D_f}")
        print(f"Difference from expected: {abs(mean_D_f - expected_D_f):.3f}")
        print(f"Mean h_q: {mean_hq:.3f}")
        print(f"Number of valid subsets: {np.sum(np.isfinite(D_f_values))}")
        p_value = validate_df(D_f_values, expected_D_f, sigma_D_f)
        print(f"Z-test p-value: {p_value:.3f}")

        # Set full per-subset log here (main thread, after parallel)
        metadata_log['subsets_processed'][dataset_name] = {
            'n_subsets': n_subsets,
            'D_f_values': D_f_values,
            'hq_values': [hq.tolist() if np.any(np.isfinite(hq)) else [] for hq in hq_values],
            'fluct_values': [fluct.tolist() if np.any(np.isfinite(fluct)) else [] for fluct in fluct_values],
            'slopes_values': [slopes.tolist() if np.any(np.isfinite(slopes)) else [] for slopes in slopes_values]
        }
        # Log dataset-level summary
        metadata_log['subsets_processed'][dataset_name].update({
            'mean_D_f': mean_D_f, 'std_D_f': std_D_f, 'mean_hq': mean_hq, 'p_value': p_value,
            'expected_D_f': expected_D_f, 'sigma_D_f': sigma_D_f, 'n_valid_subsets': np.sum(np.isfinite(D_f_values))
        })

        return {
            'D_f': D_f_values, 'hq_mean': [np.nanmean(hq) for hq in hq_values], 'fluct': fluct_values,
            'hq_values': hq_values,       
            'hq_mean': hq_means,
            'slopes': slopes_values,  # Added for completeness
            'mean_D_f': mean_D_f, 'std_D_f': std_D_f, 'mean_hq': mean_hq, 'p_value': p_value,
            'best_subsets': subset_data_list
        }, subset_data_list
    else:
        print("No valid D_f values computed.")
        metadata_log['subsets_processed'][dataset_name] = {  # Set empty for failed datasets
            'n_subsets': n_subsets, 'D_f_values': D_f_values, 'hq_values': hq_values,
            'fluct_values': fluct_values, 'slopes_values': slopes_values
        }
        metadata_log['subsets_processed'][dataset_name].update({
            'mean_D_f': np.nan, 'std_D_f': np.nan, 'mean_hq': np.nan, 'p_value': np.nan,
            'expected_D_f': expected_D_f, 'sigma_D_f': sigma_D_f, 'n_valid_subsets': 0
        })
        metadata_log['errors'].append({'dataset': dataset_name, 'error': 'No valid D_f'})
        return None, []

def convert_numpy(obj):
    """
    Recursive convert NumPy-types to Python-natives for JSON-serialisatie.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    else:
        return obj

import copy

def save_run_metadata(save_flag):
    if not save_flag:
        return

    # 1. Light CSV 
    df = pd.DataFrame(columns=['type', 'pair', 'dataset', 'subset_idx', 'D_f', 'hq_mean',
                               'corr', 'wavelet_corr', 'weight', 'tci_global', 'mci_global',
                               'fold_idx', 'tmci_value', 'tmci_global', 'tmci_std_cv'])

    # TCI
    for pair, d in metadata_log.get('tci_pairs', {}).items():
        df = pd.concat([df, pd.DataFrame([{
            'type': 'TCI', 'pair': pair, 'corr': d.get('corr'),
            'wavelet_corr': d.get('wavelet_corr'), 'weight': d.get('weight')
        }])], ignore_index=True)

    # MCI
    for item in metadata_log.get('mci_measurements', []):
        df = pd.concat([df, pd.DataFrame([{
            'type': 'MCI', 'pair_idx': item.get('pair_idx'),
            'corr': item.get('corr'), 'psd_corr': item.get('psd_corr')
        }])], ignore_index=True)

    # TMCI folds
    for i, val in enumerate(metadata_log.get('tmci_folds', [])):
        df = pd.concat([df, pd.DataFrame([{
            'type': 'TMCI_fold', 'fold_idx': i, 'tmci_value': val
        }])], ignore_index=True)

    # Max number of subsets
    subset_counter = 0
    for dataset, proc in metadata_log.get('subsets_processed', {}).items():
        for i in range(len(proc.get('D_f_values', []))):
            if subset_counter >= 5000:
                break
            hq_mean = np.nanmean(proc['hq_values'][i]) if i < len(proc['hq_values']) else np.nan
            df = pd.concat([df, pd.DataFrame([{
                'type': 'MFDFA_subset', 'dataset': dataset, 'subset_idx': i,
                'D_f': proc['D_f_values'][i], 'hq_mean': hq_mean
            }])], ignore_index=True)
            subset_counter += 1
        if subset_counter >= 5000:
            break

    # Global metrics
    df = pd.concat([df, pd.DataFrame([{
        'type': 'GLOBAL',
        'tci_global': metadata_log.get('tci'),
        'mci_global': metadata_log.get('mci'),
        'tmci_global': metadata_log.get('tmci'),
        'tmci_std_cv': metadata_log.get('tmci_std_cv')
    }])], ignore_index=True)

    timestamp = metadata_log['run_timestamp']
    csv_path = f"{OUTPUT_DIR}full_melted_utmf_v5.0_cell_1_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"→ Light CSV saved ({len(df)} rows): {csv_path}")

# Calculate TCI (Temporal Coherence Index)
def calculate_tci_multivariate(tci_datasets):
    spectra = []
    valid_names = []

    for name, series in tci_datasets.items():
        if series is None or len(series) < 2048:
            continue
        if isinstance(series, np.ndarray) and series.ndim > 1:
            series = np.mean(series, axis=1)
        series = series[:131072]

        try:
            
            scales = np.logspace(np.log10(8), np.log10(len(series)//8), 15).astype(int)
            scales = scales[scales > 4]
            q_values = CONFIG['mfdfa']['q_values']

            D_f, hq, _, _, _ = jedi_mfdfa(series, scales, q_values)

            if np.any(np.isnan(hq)):
                # simple lineair interpolation of valid points
                valid = np.isfinite(hq)
                if np.sum(valid) < 2:
                    continue
                hq = np.interp(q_values, q_values[valid], hq[valid])

            spectra.append(hq)
            valid_names.append(name)
            print(f"[TCI-fractal] {name}: h(q) succes")
        except Exception as e:
            continue

    if len(spectra) < 2:
        print(f"[TCI-fractal] No sufficient spectra ({len(spectra)}) → TCI = 0.0")
        return 0.0, {}

    all_corrs = []
    for i in range(len(spectra)):
        for j in range(i+1, len(spectra)):
            corr, _ = pearsonr(spectra[i], spectra[j])
            all_corrs.append(abs(corr))

    tci = np.mean(all_corrs)
    print(f"[TCI-fractal] Fractal TCI = {tci:.3f} ({len(all_corrs)} pairs from {len(valid_names)} datasets)")
    return tci, {}

# Calculate MCI (Measurement Coherence Index)
def calculate_mci_multivariate(results_all):
    spectra = []
    valid_datasets = []

    for dataset_name, res in results_all.items():
        if isinstance(res, tuple) and len(res) > 0:
            res = res[0]  
        if not isinstance(res, dict):
            continue
        if 'hq_values' not in res:
            continue

        hq_list = res['hq_values']
        if len(hq_list) == 0:
            continue

        try:
            hq_array = np.stack(hq_list)               # (n_subsets, n_q)
            mean_hq = np.nanmean(hq_array, axis=0)
            if len(mean_hq) < 5 or np.all(np.isnan(mean_hq)):
                continue
            spectra.append(mean_hq)
            valid_datasets.append(dataset_name)
        except Exception as e:
            continue  

    print(f"[MCI-4.3] Datasets with h(q): {len(valid_datasets)}")

    if len(spectra) < 2:
        print("[MCI-4.3] Less than 2 valid spectra → MCI = 0.0")
        return 0.0

    all_corrs = []
    for i in range(len(spectra)):
        for j in range(i + 1, len(spectra)):
            min_len = min(len(spectra[i]), len(spectra[j]))
            corr, _ = pearsonr(spectra[i][:min_len], spectra[j][:min_len])
            all_corrs.append(abs(corr))

    mci = np.nanmean(all_corrs)
    print(f"[MCI-4.3] Multifractal MCI = {mci:.3f} ({len(all_corrs)} pairs from {len(valid_datasets)} datasets)")
    return mci

# Calculate TMCI (Temporal-Measurement Coherence Index)
def calculate_tmci_empirical(tci, mci, n_boot=500):
    """
    Empirical TMCI:
    - bootstrap 
    - correlation between TCI en MCI
    - statistical weighted index (data-driven)
    """

    if np.isnan(tci) or np.isnan(mci):
        return np.nan, np.nan, np.nan

    # bootstrap samples simulation via jitter
    tci_samples = tci + np.random.normal(0, 0.02, n_boot)
    mci_samples = mci + np.random.normal(0, 0.02, n_boot)

    # correlation between TCI en MCI
    corr_tm, _ = pearsonr(tci_samples, mci_samples)

    # data-driven weights
    # the more they correlate, the more 1 component is sufficient
    w = (1 - abs(corr_tm)) / 2       # 0 = completely redundant; 0.5 = completely independent

    tmci_samples = w * tci_samples + w * mci_samples + (1 - 2*w) * ((tci_samples + mci_samples) / 2)

    tmci_mean = np.mean(tmci_samples)
    tmci_ci = (np.percentile(tmci_samples, 2.5), np.percentile(tmci_samples, 97.5))

    return tmci_mean, tmci_ci, corr_tm

# Load datasets
def load_utmf_data():
    results_all = {}
    tci_datasets = {}
    mci_datasets = {}
    metadata_log['config_snapshot'] = CONFIG.copy() # Snapshot

    # LIGO
    for idx, (file_path, dataset_name) in enumerate(zip(CONFIG['ligo_files'], CONFIG['ligo_names'])):
        if CONFIG['ligo'][idx]['utmf_use']:
            print(f"\n[UTMF] Loading LIGO: {dataset_name}")
            for_mfdfa, tci_extracted, mci_extracted = load_data(file_path, 'ligo', ligo_idx=idx)
            if for_mfdfa is not None:
                results, _ = process_dataset(for_mfdfa, 'ligo', dataset_name, CONFIG['ligo'][idx]['scales'],
                                             CONFIG['ligo'][idx]['expected_D_f'], CONFIG['ligo'][idx]['sigma_D_f'], ligo_idx=idx)
                if results:
                    results_all[dataset_name] = results
                    tci_datasets[dataset_name] = tci_extracted
                    mci_datasets[dataset_name] = mci_extracted
                    metadata_log['datasets_loaded'].append({'name': dataset_name, 'n_subsets': CONFIG['ligo'][idx]['n_subsets'], 'status': 'OK'})
    gc.collect()

    # CMB/Planck
    for idx, (file_path, dataset_name) in enumerate(zip(CONFIG['cmb_files'], CONFIG['cmb_names'])):
        if CONFIG['cmb'][idx]['utmf_use']:
            print(f"\n[UTMF] Loading CMB: {dataset_name}")
            for_mfdfa, tci_extracted, mci_extracted = load_data(file_path, 'cmb', cmb_idx=idx)
            if for_mfdfa is not None:
                results, _ = process_dataset(for_mfdfa, 'cmb', dataset_name, CONFIG['cmb'][idx]['scales'],
                                             CONFIG['cmb'][idx]['expected_D_f'], CONFIG['cmb'][idx]['sigma_D_f'], cmb_idx=idx)
                if results:
                    results_all[dataset_name] = results
                    tci_datasets[dataset_name] = tci_extracted
                    mci_datasets[dataset_name] = mci_extracted
                    metadata_log['datasets_loaded'].append({'name': dataset_name, 'n_subsets': CONFIG['cmb'][idx]['n_subsets'], 'status': 'OK'})
    gc.collect()

    # DESI
    if CONFIG['desi'].get('utmf_use', False):
        print(f"\n[UTMF] Loading DESI: {CONFIG['desi_name']}")
        for_mfdfa, tci_extracted, mci_extracted = load_data(CONFIG['desi_file'], 'desi')
        if for_mfdfa is not None:
            results, _ = process_dataset(for_mfdfa, 'desi', CONFIG['desi_name'], CONFIG['desi']['scales'],
                                         CONFIG['desi']['expected_D_f'], CONFIG['desi']['sigma_D_f'])
            if results:
                results_all[CONFIG['desi_name']] = results
                tci_datasets[CONFIG['desi_name']] = tci_extracted
                mci_datasets[CONFIG['desi_name']] = mci_extracted
                metadata_log['datasets_loaded'].append({'name': CONFIG['desi_name'], 'n_subsets': CONFIG['desi']['n_subsets'], 'status': 'OK'})
    gc.collect()

    # CERN
    if CONFIG['cern'].get('utmf_use', False):
        print(f"\n[UTMF] Loading CERN: {CONFIG['cern_name']}")
        for_mfdfa, tci_extracted, mci_extracted = load_data(CONFIG['cern_file'], 'cern')
        if for_mfdfa is not None:
            results, _ = process_dataset(for_mfdfa, 'cern', CONFIG['cern_name'], CONFIG['cern']['scales'],
                                         CONFIG['cern']['expected_D_f'], CONFIG['cern']['sigma_D_f'])
            if results:
                results_all[CONFIG['cern_name']] = results
                tci_datasets[CONFIG['cern_name']] = tci_extracted
                mci_datasets[CONFIG['cern_name']] = mci_extracted
                metadata_log['datasets_loaded'].append({'name': CONFIG['cern_name'], 'n_subsets': CONFIG['cern']['n_subsets'], 'status': 'OK'})
    gc.collect()

    # NIST (flagged elements)
    if CONFIG['nist'].get('utmf_use', False):
        for elements in CONFIG['nist']['elements_list_utmf']:
            elements_str = '_'.join(elements)
            full_element_name = ELEMENT_NAMES.get(elements[0], elements[0])
            dataset_name = f"NIST_3_{elements_str}_{full_element_name}"
            CONFIG['nist']['elements'] = elements
            print(f"\n[UTMF] Loading NIST: {dataset_name}")
            load_result = load_data(CONFIG['nist_file'], 'nist')
            if len(load_result) == 4:
                for_mfdfa, tci_extracted, mci_extracted, full_element_returned = load_result
            else:
                for_mfdfa, tci_extracted, mci_extracted = load_result
                full_element_returned = None
            if for_mfdfa is not None:
                results, _ = process_dataset(for_mfdfa, 'nist', dataset_name, CONFIG['nist']['scales'](len(for_mfdfa)),
                                            CONFIG['nist']['expected_D_f'], CONFIG['nist']['sigma_D_f'])
                if results:
                    results_all[dataset_name] = results
                    tci_datasets[dataset_name] = tci_extracted
                    mci_datasets[dataset_name] = mci_extracted
                    results['full_element'] = full_element_returned
                    metadata_log['datasets_loaded'].append({'name': dataset_name, 'n_subsets': CONFIG['nist']['n_subsets'], 'status': 'OK'})
    gc.collect()

    # NANOGrav (flagged pulsars)
    def process_utmf_pulsar(pulsar_name):
        if pulsar_name in CONFIG['nanograv']['pulsar_list_utmf'] and CONFIG['nanograv'].get('utmf_use', False):
            dataset_name = f"NANOGrav_{pulsar_name}"
            print(f"[UTMF] Loading NANOGrav: {dataset_name}")
            for_mfdfa, tci_extracted, mci_extracted = load_data(None, 'nanograv', pulsar_name=pulsar_name)
            if for_mfdfa is not None:
                results, _ = process_dataset(for_mfdfa, 'nanograv', dataset_name, CONFIG['nanograv']['scales'],
                                             CONFIG['nanograv']['expected_D_f'], CONFIG['nanograv']['sigma_D_f'])
                if results:
                    return dataset_name, results, tci_extracted, mci_extracted, {
                        'pulsar': pulsar_name, 'mean_D_f': results['mean_D_f'], 'std_D_f': results['std_D_f'],
                        'mean_hq': results['mean_hq'], 'p_value': results['p_value']
                    }
        return None, None, None, None, None

    pulsar_results = Parallel(n_jobs=-1)(delayed(process_utmf_pulsar)(pulsar_name) for pulsar_name in tqdm(CONFIG['nanograv']['pulsar_list_utmf'], desc="[UTMF] Pulsars"))
    pulsar_summaries = []
    for ds_name, results, tci_ex, mci_ex, summary in pulsar_results:
        if results:
            results_all[ds_name] = results
            tci_datasets[ds_name] = tci_ex
            mci_datasets[ds_name] = mci_ex
            if summary:
                pulsar_summaries.append(summary)
                metadata_log['datasets_loaded'].append({'name': ds_name, 'n_subsets': CONFIG['nanograv']['n_subsets'], 'status': 'OK'})
    gc.collect()

    # Summary of pulsar results (console output)
    print("\nSummary of MFDFA results for NANOGrav pulsars:")
    for summ in pulsar_summaries:
        print(f"Pulsar {summ['pulsar']}:")
        print(f"  Mean D_f: {summ['mean_D_f']:.3f} ± {summ['std_D_f']:.3f}")
        print(f"  Mean h_q: {summ['mean_hq']:.3f}")
        print(f"  P-value: {summ['p_value']:.3f}")

    # QRNG
    if CONFIG['qrng'].get('utmf_use', False):
        print(f"\n[UTMF] Loading QRNG: NIST_QRNG")
        for_mfdfa, tci_extracted, mci_extracted = load_data(None, 'qrng')
        if for_mfdfa is not None:
            dataset_name = 'NIST_QRNG'
            results, _ = process_dataset(for_mfdfa, 'qrng', dataset_name, CONFIG['qrng']['scales'],
                                         CONFIG['qrng']['expected_D_f'], CONFIG['qrng']['sigma_D_f'])
            if results:
                results_all[dataset_name] = results
                tci_datasets[dataset_name] = tci_extracted
                mci_datasets[dataset_name] = mci_extracted
                metadata_log['datasets_loaded'].append({'name': dataset_name, 'n_subsets': CONFIG['qrng']['n_subsets'], 'status': 'OK'})
    gc.collect()

    # Gaia DR3
    if CONFIG['gaia'].get('utmf_use', False):
        print(f"\n[UTMF] Loading Gaia: {CONFIG['gaia']['name']}")
        for_mfdfa, tci_extracted, mci_extracted = load_data(CONFIG['gaia']['file'], 'gaia')
        if for_mfdfa is not None:
            dataset_name = CONFIG['gaia']['name']
            results, _ = process_dataset(for_mfdfa, 'gaia', dataset_name, CONFIG['gaia']['scales'],
                                        CONFIG['gaia']['expected_D_f'], CONFIG['gaia']['sigma_D_f'])
            if results:
                results_all[dataset_name] = results
                tci_datasets[dataset_name] = tci_extracted
                mci_datasets[dataset_name] = mci_extracted
                metadata_log['datasets_loaded'].append({'name': dataset_name, 'n_subsets': CONFIG['gaia']['n_subsets'], 'status': 'OK'})
    gc.collect()

# TCI/MCI/TMCI if enabled
    if CONFIG.get('compute_indices', False):
        print("\n[UTMF] Computing indices...")
        # TCI/MCI/TMCI
        tci, tci_pairs = calculate_tci_multivariate(tci_datasets)
        mci = calculate_mci_multivariate(results_all)

        # TMCI
        tmci_mean, tmci_ci, tmci_corr = calculate_tmci_empirical(tci, mci)
        tmci = tmci_mean   

        # Logging
        metadata_log['tci'] = tci
        metadata_log['mci'] = mci
        metadata_log['tmci'] = tmci
        metadata_log['tmci_ci'] = tmci_ci
        metadata_log['tmci_corr'] = tmci_corr

        print(f"TCI: {tci:.3f}")
        print(f"MCI: {mci:.3f}")
        print(f"TMCI(avg): {tmci:.3f}   95% CI = [{tmci_ci[0]:.3f}, {tmci_ci[1]:.3f}]")
        print(f"TCI–MCI corr: {tmci_corr:.3f}")


        # Optional cross-val (simple k=3 on tci/mci data for TMCI std)
        if CONFIG.get('cross_val', False):
            print("\n[UTMF] Running cross-val (k=3) on fractal TCI/MCI...")
            tmci_folds = []
            for fold in range(3):
                # Simuleer fold by removing 1/3 of subsets per dataset
                fold_results = {}
                for name, res in results_all.items():
                    if isinstance(res, tuple):
                        res = res[0]
                    if not isinstance(res, dict) or 'hq_values' not in res:
                        continue
                    hq_list = res['hq_values']
                    # Take random 2/3 from the subsets
                    idx = np.random.choice(len(hq_list), size=int(len(hq_list)*2/3), replace=False)
                    folded_hq = np.stack([hq_list[i] for i in idx])
                    mean_hq = np.nanmean(folded_hq, axis=0)
                    fold_results[name] = {'hq_values': [mean_hq]}  # fake list with 1 element

                # Calculate TCI/MCI on this fold
                tci_fold = calculate_tci_multivariate({name: tci_datasets[name] for name in fold_results.keys()})[0]
                mci_fold = calculate_mci_multivariate(fold_results)
                tmci_f_mean, tmci_f_ci, tmci_f_corr = calculate_tmci_empirical(tci_fold, mci_fold)
                tmci_fold = tmci_f_mean      # consistent with main TMCI
                tmci_folds.append(tmci_fold)

                print(f"Fold {fold+1}: TMCI = {tmci_fold:.3f}   (CI = [{tmci_f_ci[0]:.3f}, {tmci_f_ci[1]:.3f}])")


            tmci_std = np.std(tmci_folds) if tmci_folds else 0.0
            print(f"TMCI cross-val std: {tmci_std:.3f}")
            metadata_log['tmci_std_cv'] = tmci_std
        if tmci_folds:
            tmci_std = np.nanstd(tmci_folds)
            metadata_log['tmci_std_cv'] = tmci_std
            metadata_log['tmci_folds'] = tmci_folds
            print(f"TMCI cross-val std: {tmci_std:.3f} (p <0.05)")
        else:
            print("No valid folds.")

    # Globals
    globals().update({
        'utmf_results_all': results_all,
        'utmf_tci_datasets': tci_datasets,
        'utmf_mci_datasets': mci_datasets
    })
    print(f"[UTMF] Loaded {len(results_all)} datasets.")
    return results_all, tci_datasets, mci_datasets

# Run
results_all, tci_datasets, mci_datasets = load_utmf_data()
save_run_metadata(CONFIG['metadata']['save_flag'])

# =============================================
# Full JSON – ONLY if SAVE_FULL_DETAILS_JSON = True
# =============================================
if SAVE_FULL_DETAILS_JSON:
    json_path = f"{OUTPUT_DIR}FULL_DETAILS_utmf_v5.0_cell_1_{timestamp}.json"

    # deep copy + remove functions/lambdas
    safe_log = copy.deepcopy(metadata_log)
    safe_config = {k: v for k, v in CONFIG.items() if not callable(v)}
    if 'scales' in safe_config.get('nist', {}):
        safe_config['nist']['scales'] = "lambda removed for JSON"
    if 'subset_size' in safe_config.get('nist', {}):
        safe_config['nist']['subset_size'] = "lambda removed for JSON"
    safe_log['config_snapshot'] = safe_config

    # Recursive numpy → python
    safe_log = convert_numpy(safe_log)

    with open(json_path, 'w') as f:
        json.dump(safe_log, f, indent=2, default=str)
    size_mb = os.path.getsize(json_path) / 1e6
    print(f"→ Full JSON saved ({size_mb:.1f} MB): {json_path}")
else:
    print("→ Full JSON skipped (SAVE_FULL_DETAILS_JSON = False)")

mci_val = locals().get('mci', metadata_log.get('mci', np.nan))
tci_val = locals().get('tci', metadata_log.get('tci', np.nan))
tmci_val = locals().get('tmci', metadata_log.get('tmci', np.nan))
tmci_std_val = locals().get('tmci_std', metadata_log.get('tmci_std_cv', np.nan))
tmci_folds_val = locals().get('tmci_folds', metadata_log.get('tmci_folds', []))
tci_pairs_val = locals().get('tci_pairs', metadata_log.get('tci_pairs', {}))

globals().update({
    'utmf_results_all_full': results_all,   # Per dataset: full D_f, hq, fluct lists
    'utmf_tci_datasets_full': tci_datasets, # TCI
    'utmf_mci_datasets_full': mci_datasets, # MCI
    'utmf_config_snapshot': CONFIG.copy(),  # All params, scales, q_values, etc.
    'utmf_metadata_log': metadata_log,      # Full log: subsets_processed (per-subset D_f/hq/fluct/slopes), errors, tci_pairs (per-pair corr/wavelet/weight), mci_measurements (per-corr), tmci_folds
    'utmf_tci_pairs_full': tci_pairs_val,      # Dict of pairs with details
    'utmf_mci_measurements_full': metadata_log.get('mci_measurements', []),  # List of all MCI corrs
    'utmf_mci_full': mci_val,
    'utmf_tci_full': tci_val,
    'utmf_tmci_full': tmci_val,
    'utmf_tmci_std_cv_full': tmci_std_val,
    'utmf_tmci_folds_full': tmci_folds_val
})

print("Globals updated: Everything passed for Cel 2 (incl. per-subset MFDFA, full TCI/MCI details, params)!")

print(f"TMCI = {tmci_val:.3f}, MCI = {mci_val:.3f}")
