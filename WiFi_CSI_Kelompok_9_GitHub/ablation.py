"""Bandingkan MLP dengan vs tanpa Butterworth, split dan parameter identik."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from src.config import ABLATION_SEED, EPOCHS
from src.dataset import load_prepared, normalize_splits
from src.preprocessing import build_dataset
from src.engine import set_seed, make_loader, fit_model, predict
from src.models import MLP
from evaluate import metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/raw')
    p.add_argument('--output-dir', default='artifacts')
    p.add_argument('--epochs', type=int, default=EPOCHS)
    args = p.parse_args()
    out = Path(args.output_dir)
    prepared = load_prepared(out)
    (out / 'models').mkdir(parents=True, exist_ok=True)
    (out / 'results').mkdir(parents=True, exist_ok=True)
    comparisons, histories = [], {}
    true_y = prepared['y_test']

    for use_filter, description, suffix in [
        (True, 'Dengan Butterworth', 'dengan_filter'),
        (False, 'Tanpa Butterworth', 'tanpa_filter')
    ]:
        X, y, groups = build_dataset(args.data_dir, butterworth=use_filter)
        train_mask = np.isin(groups, prepared['train_ids'])
        val_mask = np.isin(groups, prepared['val_ids'])
        test_mask = np.isin(groups, prepared['test_ids'])
        targets = [y[mask] for mask in (train_mask, val_mask, test_mask)]
        if not all(len(targets[i]) == len(prepared[key]) and
                   np.array_equal(targets[i], prepared[key])
                   for i, key in enumerate(('y_train', 'y_val', 'y_test'))):
            raise ValueError('Split/urutan dataset ablation berbeda dari eksperimen utama!')
        (xtr, xva, xte), _, _ = normalize_splits(
            X[train_mask], X[val_mask], X[test_mask])
        set_seed(ABLATION_SEED)
        generator = torch.Generator().manual_seed(ABLATION_SEED)
        tr_loader = make_loader(xtr, targets[0], shuffle=True, generator=generator)
        va_loader = make_loader(xva, targets[1])
        model, history = fit_model(MLP(), tr_loader, va_loader, epochs=args.epochs,
                                   label=description)
        checkpoint = out / 'models' / f'ablation_mlp_{suffix}.pt'
        torch.save(model.state_dict(), checkpoint)
        pred = predict(model, xte)
        comparisons.append({'Perlakuan': description, **metrics(true_y, pred)})
        histories[description] = history
        print('Checkpoint disimpan:', checkpoint)
    table = pd.DataFrame(comparisons)
    table.to_csv(out / 'results' / 'ablation_mlp_filter.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, history in histories.items():
        ax.plot(history['val'], label=name)
    ax.set(xlabel='Epoch', ylabel='SmoothL1 validation loss',
           title='Ablation: Pengaruh Butterworth pada MLP')
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out / 'results' / 'ablation_validation_loss.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(table['Perlakuan'], table['Mean error (grid)'])
    ax.set(ylabel='Mean error (grid; lebih kecil lebih baik)',
           title='Ablation: Mean Distance Error')
    fig.tight_layout()
    fig.savefig(out / 'results' / 'ablation_mean_error.png', dpi=160)
    plt.close(fig)
    print('\nHASIL ABLATION STUDY:\n', table.to_string(index=False))


if __name__ == '__main__':
    main()
