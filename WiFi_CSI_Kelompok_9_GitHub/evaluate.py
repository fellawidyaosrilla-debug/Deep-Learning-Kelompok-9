"""Evaluasi regresi 2D dan evaluasi zona turunan serta confusion matrix."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_fscore_support, classification_report)
from src.config import ZONE_NAMES
from src.dataset import load_prepared, coords_to_zone
from src.engine import device, load_weights, predict
from src.models import make_model


def metrics(y_true, y_pred):
    dist = np.linalg.norm(y_pred - y_true, axis=1)
    return {
        'Mean error (grid)': float(dist.mean()),
        'Median error (grid)': float(np.median(dist)),
        'P90 error (grid)': float(np.quantile(dist, .9))
    }


def main():
    p = argparse.ArgumentParser(description='Evaluasi MLP & CNN-LSTM pada data testing')
    p.add_argument('--output-dir', default='artifacts')
    p.add_argument('--model-dir', help='Folder .pt. Misalnya folder models di Google Drive')
    args = p.parse_args()
    out = Path(args.output_dir)
    models_dir = Path(args.model_dir) if args.model_dir else out / 'models'
    results_dir = out / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    prepared = load_prepared(out)
    X_test, y_test = prepared['X_test'], prepared['y_test']
    actual_zones = coords_to_zone(y_test)
    models = [('MLP', 'MLP.pt', 'mlp'), ('CNN-LSTM', 'CNN_LSTM.pt', 'cnn_lstm')]
    comparisons, zone_results, predictions = [], [], {}
    print('Evaluasi memakai', len(y_test), 'window testing. Satuan: grid, bukan meter.')
    print('GPU/CPU:', device())
    for name, checkpoint, suffix in models:
        path = models_dir / checkpoint
        if not path.exists():
            raise FileNotFoundError(f'Checkpoint tidak ditemukan: {path}. '
                                    'Jalankan train.py atau gunakan --model-dir')
        model = load_weights(make_model(name), path)
        pred = predict(model, X_test)
        predictions[name] = pred
        comparisons.append({'Model': name, **metrics(y_test, pred)})
        zone_pred = coords_to_zone(pred)
        cm = confusion_matrix(actual_zones, zone_pred, labels=[0, 1, 2, 3])
        pd.DataFrame(cm, index=[f'Aktual {z}' for z in ZONE_NAMES],
                     columns=[f'Prediksi {z}' for z in ZONE_NAMES]).to_csv(
                         results_dir / f'confusion_matrix_{suffix}.csv')
        fig, ax = plt.subplots(figsize=(6.4, 5.5))
        ConfusionMatrixDisplay(cm, display_labels=list(ZONE_NAMES)).plot(
            ax=ax, cmap='Blues', colorbar=False, values_format='d', xticks_rotation=25)
        ax.set(title=f'Confusion Matrix Zona - {name}',
               xlabel='Zona prediksi', ylabel='Zona sebenarnya')
        fig.tight_layout()
        fig.savefig(results_dir / f'confusion_matrix_{suffix}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        prec, recall, f1, _ = precision_recall_fscore_support(
            actual_zones, zone_pred, labels=[0, 1, 2, 3], average='macro', zero_division=0)
        zone_results.append({
            'Model': name, 'Total sampel testing': len(y_test),
            'Akurasi zona (%)': 100 * accuracy_score(actual_zones, zone_pred),
            'Precision macro': float(prec), 'Recall macro': float(recall), 'F1 macro': float(f1)
        })
        print(f'\n=== {name} ===')
        print(classification_report(actual_zones, zone_pred,
                                    labels=[0, 1, 2, 3], target_names=ZONE_NAMES,
                                    zero_division=0))
    df = pd.DataFrame(comparisons).sort_values('Mean error (grid)')
    df.to_csv(results_dir / 'model_comparison.csv', index=False)
    pd.DataFrame(zone_results).to_csv(results_dir / 'zone_classification_comparison.csv', index=False)
    np.savez_compressed(results_dir / 'predictions.npz',
                        mlp=predictions['MLP'], cnn_lstm=predictions['CNN-LSTM'], actual=y_test)
    print('\nPERBANDINGAN MODEL:\n', df.to_string(index=False))
    print('\nKLASIFIKASI ZONA:\n', pd.DataFrame(zone_results).to_string(index=False))
    best = df.iloc[0]['Model']
    pred_best = predictions[best]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test[:, 0], y_test[:, 1], label='Lokasi sebenarnya', alpha=.8)
    ax.scatter(pred_best[:, 0], pred_best[:, 1], label='Prediksi ' + best, marker='x', alpha=.8)
    for actual, pred in zip(y_test[:35], pred_best[:35]):
        ax.plot([actual[0], pred[0]], [actual[1], pred[1]], alpha=.2, linewidth=.7)
    ax.set(xlabel='X (grid)', ylabel='Y (grid)', title='Prediksi vs koordinat sebenarnya')
    ax.grid(alpha=.25)
    ax.axis('equal')
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / 'localization_scatter.png', dpi=160)
    plt.close(fig)
    print('\nFile hasil tersimpan:', results_dir)


if __name__ == '__main__':
    main()
