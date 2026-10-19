"""Unduh subset contoh dari repositori dataset asli (tidak diunggah ulang)."""
import argparse
from pathlib import Path
import shutil
import subprocess

URL = 'https://github.com/zhangleino1/WIFI-SCI-Indoor-Positioning.git'


def main():
    parser = argparse.ArgumentParser(description='Unduh 270 CSV CSI dari GitHub sumber')
    parser.add_argument('--data-dir', default='data/raw')
    args = parser.parse_args()
    target = Path(args.data_dir)
    existing = list(target.glob('antenna_*.csv')) if target.exists() else []
    if len(existing) >= 270:
        print(f'Sudah tersedia {len(existing)} file CSI di {target}. Lewati unduh.')
        return
    source = Path('data/upstream-source')
    if not (source / 'dataset').exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(['git', 'clone', '--depth', '1', URL, str(source)], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise SystemExit('Gagal mengunduh. Pastikan Git terinstal dan internet aktif.\n'
                             f'Repositori: {URL}\nDetail: {exc}') from exc
    csvs = sorted((source / 'dataset').glob('antenna_*.csv'))
    if not csvs:
        raise SystemExit('CSV antenna_*.csv tidak ditemukan dalam folder dataset di upstream.')
    target.mkdir(parents=True, exist_ok=True)
    for f in csvs:
        shutil.copy2(f, target / f.name)
    print(f'Berhasil: {len(csvs)} file CSI tersimpan di {target}')
    print('Sumber asli:', URL)


if __name__ == '__main__':
    main()
