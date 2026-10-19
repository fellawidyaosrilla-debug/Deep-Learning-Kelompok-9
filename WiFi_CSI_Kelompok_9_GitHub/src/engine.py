"""Training/evaluasi berbasis PyTorch, validasi & early stopping."""
import copy
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from .config import SEED, BATCH_SIZE, PATIENCE, LEARNING_RATE


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_loader(X, y, shuffle=False, batch_size=BATCH_SIZE, generator=None):
    return DataLoader(TensorDataset(torch.from_numpy(np.asarray(X, dtype=np.float32)),
                                    torch.from_numpy(np.asarray(y, dtype=np.float32))),
                      batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=generator)


def val_loss(model, loader, loss_fn, dev):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            total += loss_fn(model(xb), yb).item() * len(xb)
    return total / len(loader.dataset)


def fit_model(model, train_loader, val_loader, epochs=20, dev=None,
              patience=PATIENCE, label='Model'):
    dev = dev or device()
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.SmoothL1Loss()
    best = float('inf')
    best_weights = None
    wait = 0
    hist = {'train': [], 'val': []}
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        train_loss = total / len(train_loader.dataset)
        validation = val_loss(model, val_loader, loss_fn, dev)
        hist['train'].append(train_loss)
        hist['val'].append(validation)
        print(f'{label} | Epoch {epoch:02d} | train={train_loss:.4f} | val={validation:.4f}')
        if validation < best - 1e-5:
            best, wait = validation, 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                print(f'Early stopping: {label}')
                break
    if best_weights is None:
        raise RuntimeError('Training gagal: tidak ada bobot terbaik')
    model.load_state_dict(best_weights)
    return model, hist


def predict(model, X, dev=None, batch_size=BATCH_SIZE):
    dev = dev or device()
    model = model.to(dev).eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(np.asarray(X[i:i + batch_size], dtype=np.float32)).to(dev)
            outs.append(model(xb).cpu().numpy())
    if not outs:
        raise ValueError('Tidak ada sampel testing')
    return np.concatenate(outs)


def load_weights(model, path, dev=None):
    dev = dev or device()
    # weights_only tersedia di PyTorch 2.1+; checkpoint notebook berupa state_dict.
    try:
        state = torch.load(path, map_location=dev, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=dev)
    model.load_state_dict(state)
    return model.to(dev).eval()
