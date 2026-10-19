"""Split per-lokasi stratifikasi 4 zona, normalisasi tanpa data leakage."""
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from .config import SEED, X_BOUNDARY, Y_BOUNDARY
from .preprocessing import build_dataset


def coords_to_zone(points):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('Koordinat harus berbentuk (N, 2)')
    return ((points[:, 0] > X_BOUNDARY).astype(int)
            + 2 * (points[:, 1] > Y_BOUNDARY).astype(int))


def group_split(y, groups, seed=SEED):
    """70/15/15 per lokasi; sklearn akan menentukan pembulatan sesuai notebook."""
    group_ids = np.unique(groups)
    group_coords = np.array([y[np.where(groups == gid)[0][0]] for gid in group_ids])
    zones = coords_to_zone(group_coords)
    train_ids, rem_ids = train_test_split(group_ids, test_size=.30,
                                         random_state=seed, stratify=zones)
    zone_by_group = dict(zip(group_ids, zones))
    rem_zones = np.array([zone_by_group[int(g)] for g in rem_ids])
    val_ids, test_ids = train_test_split(rem_ids, test_size=.50,
                                        random_state=seed, stratify=rem_zones)
    if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
        raise RuntimeError('Data leakage: lokasi dipakai di lebih dari satu split')
    return train_ids, val_ids, test_ids


def normalize_splits(train, val, test):
    mean = train.mean(axis=(0, 2, 3), keepdims=True)
    std = train.std(axis=(0, 2, 3), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    arrays = tuple(((part - mean) / std).astype(np.float32)
                   for part in (train, val, test))
    return arrays, mean, std


def prepare(data_dir, output_dir, seed=SEED, overwrite=False):
    path = Path(output_dir) / 'prepared.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        print(f'Data terproses sudah tersedia: {path} (gunakan --force untuk membuat ulang)')
        return path

    X, y, groups = build_dataset(data_dir, butterworth=True)
    tr, va, te = group_split(y, groups, seed=seed)
    masks = [np.isin(groups, ids) for ids in (tr, va, te)]
    x_splits = [X[m] for m in masks]
    y_splits = [y[m] for m in masks]
    (X_train, X_val, X_test), mu, sigma = normalize_splits(*x_splits)
    np.savez_compressed(path, X_train=X_train, X_val=X_val, X_test=X_test,
                        y_train=y_splits[0], y_val=y_splits[1], y_test=y_splits[2],
                        mean=mu, std=sigma, train_ids=tr, val_ids=va, test_ids=te,
                        seed=np.array(seed))
    for label, targets in [('Training', y_splits[0]), ('Validation', y_splits[1]), ('Testing', y_splits[2])]:
        print(f'{label}: {len(targets)} window; distribusi zona A-D = '
              f'{np.bincount(coords_to_zone(targets), minlength=4).tolist()}')
    print('Tersimpan:', path)
    return path


def load_prepared(output_dir):
    f = Path(output_dir) / 'prepared.npz'
    if not f.exists():
        raise FileNotFoundError(f'Belum ada {f}. Jalankan: python prepare_data.py')
    return np.load(f, allow_pickle=False)
