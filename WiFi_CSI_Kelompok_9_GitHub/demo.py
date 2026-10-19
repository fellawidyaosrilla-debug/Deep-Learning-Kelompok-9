"""Demo inferensi pada satu window dari test set (bukan sinyal CSI live)."""
import argparse
from pathlib import Path
import numpy as np
from src.dataset import load_prepared
from src.engine import load_weights, predict
from src.models import make_model


def main():
    p = argparse.ArgumentParser(description='Demo prediksi koordinat pada contoh testing')
    p.add_argument('--output-dir', default='artifacts')
    p.add_argument('--model-dir', help='Lokasi checkpoint .pt, default artifacts/models')
    p.add_argument('--index', type=int, default=0)
    args = p.parse_args()
    out = Path(args.output_dir)
    mdir = Path(args.model_dir) if args.model_dir else out / 'models'
    data = load_prepared(out)
    X, y = data['X_test'], data['y_test']
    if not 0 <= args.index < len(X):
        raise SystemExit(f'Index harus 0 sampai {len(X)-1}')
    print(f'Sampel testing indeks {args.index}: koordinat asli {y[args.index].tolist()} grid')
    for name, file in [('MLP', 'MLP.pt'), ('CNN-LSTM', 'CNN_LSTM.pt')]:
        path = mdir / file
        if not path.exists():
            raise FileNotFoundError(f'Model belum ada: {path}')
        model = load_weights(make_model(name), path)
        prediction = predict(model, X[args.index:args.index+1])[0]
        distance = float(np.linalg.norm(prediction - y[args.index]))
        print(f'{name}: prediksi {prediction.round(4).tolist()} grid; error={distance:.4f} grid')
    print('Demo ini menggunakan DATA TESTING yang disimpan; bukan penangkapan WiFi langsung.')


if __name__ == '__main__':
    main()
