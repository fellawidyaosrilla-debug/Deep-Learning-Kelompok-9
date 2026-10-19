"""Melatih MLP baseline dan CNN-LSTM; menghasilkan .pt dan kurva loss."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from src.config import EPOCHS, BATCH_SIZE
from src.dataset import load_prepared
from src.engine import set_seed, make_loader, fit_model, device
from src.models import make_model


def main():
    p = argparse.ArgumentParser(description='Training dua arsitektur Deep Learning')
    p.add_argument('--output-dir', default='artifacts')
    p.add_argument('--epochs', type=int, default=EPOCHS)
    p.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)
    data = load_prepared(args.output_dir)
    tr = make_loader(data['X_train'], data['y_train'], shuffle=True, batch_size=args.batch_size)
    va = make_loader(data['X_val'], data['y_val'], batch_size=args.batch_size)
    output = Path(args.output_dir)
    (output / 'models').mkdir(parents=True, exist_ok=True)
    (output / 'results').mkdir(parents=True, exist_ok=True)
    histories = {}
    print('Perangkat:', device())
    for name in ('MLP', 'CNN-LSTM'):
        model, hist = fit_model(make_model(name), tr, va, epochs=args.epochs,
                                label=name)
        ckpt = output / 'models' / ('MLP.pt' if name == 'MLP' else 'CNN_LSTM.pt')
        torch.save(model.state_dict(), ckpt)
        histories[name] = hist
        print('Checkpoint disimpan:', ckpt)
    (output / 'results' / 'training_history.json').write_text(
        json.dumps(histories, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, hist in histories.items():
        ax.plot(hist['train'], label=f'{name} train')
        ax.plot(hist['val'], '--', label=f'{name} val')
    ax.set(xlabel='Epoch', ylabel='SmoothL1 loss',
           title='Perbandingan training dan validation loss')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'results' / 'loss_comparison.png', dpi=160)
    plt.close(fig)
    print('Training selesai. Jalankan: python evaluate.py')


if __name__ == '__main__':
    main()
