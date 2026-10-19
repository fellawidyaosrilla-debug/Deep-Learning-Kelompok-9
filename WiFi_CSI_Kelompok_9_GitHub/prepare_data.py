"""Siapkan tensor CSI, stratified group split dan normalisasi."""
import argparse
from src.dataset import prepare


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/raw')
    p.add_argument('--output-dir', default='artifacts')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--force', action='store_true', help='Buat ulang data, overwrites prepared.npz')
    args = p.parse_args()
    prepare(args.data_dir, args.output_dir, seed=args.seed, overwrite=args.force)


if __name__ == '__main__':
    main()
