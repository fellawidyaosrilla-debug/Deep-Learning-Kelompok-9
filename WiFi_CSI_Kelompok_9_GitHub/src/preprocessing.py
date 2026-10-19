"""CSI: interpolasi nilai invalid, Butterworth, sanitasi phase dan windowing.

Data upstream adalah CSV amplitude/phase, BUKAN pembacaan CSI kompleks mentah.
Satuan label (x,y) adalah grid; tidak dapat diklaim sebagai meter.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, detrend
from .config import WINDOW, STRIDE, ANTENNAS, SUBCARRIERS

PATTERN = re.compile(r'antenna_(\d+)_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)\.csv$')
AMP_COLS = [f'amplitude_{i}' for i in range(SUBCARRIERS)]
PHASE_COLS = [f'phase_{i}' for i in range(SUBCARRIERS)]
SOS = butter(3, 0.25, btype='lowpass', output='sos')


def discover_locations(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f'Folder tidak ditemukan: {data_dir}. Jalankan download_data.py.')
    locations = {}
    for f in sorted(data_dir.glob('antenna_*.csv')):
        match = PATTERN.fullmatch(f.name)
        if match:
            ant = int(match[1])
            xy = float(match[2]), float(match[3])
            locations.setdefault(xy, {})[ant] = f
    complete = sorted(xy for xy, antenna_files in locations.items()
                      if set(ANTENNAS).issubset(antenna_files))
    if len(complete) < 10:
        raise ValueError(f'Hanya {len(complete)} koordinat lengkap; butuh minimal 10.')
    print(f'File CSV ditemukan: {sum(len(a) for a in locations.values())}; '
          f'lokasi dengan 3 antena: {len(complete)}')
    return locations, complete


def read_antenna(filepath, butterworth=True):
    """Menghasilkan 2 matriks (waktu, 30 subcarrier)."""
    filepath = Path(filepath)
    df = pd.read_csv(filepath)
    required = AMP_COLS + PHASE_COLS
    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError(f'{filepath.name}: kolom hilang: {missing[:5]}')

    clean = df[required].apply(pd.to_numeric, errors='coerce')
    clean = clean.replace([np.inf, -np.inf], np.nan)
    invalid = int(clean.isna().sum().sum())
    if invalid:
        print(f'[Perbaikan CSI] {filepath.name}: {invalid} nilai invalid; '
              'interpolasi sepanjang waktu')
        clean = clean.interpolate(method='linear', axis=0, limit_direction='both')
    if clean.isna().any().any():
        raise ValueError(f'{filepath.name}: ada kolom seluruhnya invalid; interpolasi gagal')

    amp = clean[AMP_COLS].to_numpy(dtype=np.float32)
    ph = clean[PHASE_COLS].to_numpy(dtype=np.float32)
    if butterworth:
        if len(amp) < 10:
            raise ValueError(f'{filepath.name}: terlalu sedikit baris untuk Butterworth')
        amp = sosfiltfilt(SOS, amp, axis=0).astype(np.float32)
    phase_clean = detrend(np.unwrap(ph, axis=1), axis=1, type='linear').astype(np.float32)
    if not (np.isfinite(amp).all() and np.isfinite(phase_clean).all()):
        raise ValueError(f'{filepath.name}: hasil preprocessing bukan nilai finite')
    return amp, phase_clean


def build_dataset(data_dir, butterworth=True):
    """Keluaran X (N,6,15,30), y (N,2), groups (N,) sesuai notebook."""
    locations, complete = discover_locations(data_dir)
    windows, labels, groups = [], [], []
    for loc_id, (x, y) in enumerate(complete):
        channels = []
        for ant in ANTENNAS:
            amp, phase = read_antenna(locations[(x, y)][ant], butterworth=butterworth)
            channels.extend((amp, phase))
        n_time = min(len(ch) for ch in channels)
        if n_time < WINDOW:
            continue
        cube = np.stack([ch[:n_time] for ch in channels], axis=0)
        for start in range(0, n_time - WINDOW + 1, STRIDE):
            windows.append(cube[:, start:start + WINDOW, :])
            labels.append((x, y))
            groups.append(loc_id)
    if not windows:
        raise ValueError('Tidak ada window; periksa dataset dan panjang rekaman')
    X = np.stack(windows).astype(np.float32)
    y = np.asarray(labels, dtype=np.float32)
    group_ids = np.asarray(groups, dtype=np.int64)
    print(f'Dataset siap: X={X.shape}, y={y.shape}, lokasi={len(np.unique(group_ids))}')
    return X, y, group_ids
