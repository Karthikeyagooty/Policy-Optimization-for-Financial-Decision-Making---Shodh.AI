import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import itertools
import time

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
MODEL_DIR = PROJECT_ROOT / 'models'
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

def load_splits():
    if (DATA_DIR / 'train.parquet').exists() and (DATA_DIR / 'test.parquet').exists():
        train = pd.read_parquet(DATA_DIR / 'train.parquet')
        val = pd.read_parquet(DATA_DIR / 'val.parquet')
        test = pd.read_parquet(DATA_DIR / 'test.parquet')
    else:
        sample = DATA_DIR / 'processed_sample.parquet'
        assert sample.exists(), 'No processed data found. Run 1_EDA.ipynb first.'
        df = pd.read_parquet(sample)
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
        train, val = train_test_split(train, test_size=0.125, random_state=42, stratify=train['target'])
    return train, val, test

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256,128), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

def evaluate(loader, model):
    model.eval()
    ys = []
    ps = []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(DEVICE).float()
            logits = model(Xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            ps.append(probs)
            ys.append(yb.numpy())
    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    auc = roc_auc_score(ys, ps) if len(np.unique(ys))>1 else float('nan')
    preds = (ps >= 0.5).astype(int)
    f1 = f1_score(ys, preds)
    return auc, f1

def train_and_evaluate(config, X_train, y_train, X_val, y_val, X_test, y_test):
    # unpack config
    hidden = tuple(config['hidden_dims'])
    dropout = config['dropout']
    lr = config['lr']
    batch_size = config['batch_size']
    weight_decay = config.get('weight_decay', 1e-5)
    epochs = config.get('epochs', 10)
    patience = config.get('patience', 3)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=1024)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=1024)

    model = MLP(input_dim=X_train.shape[1], hidden_dims=hidden, dropout=dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss()

    best_val_auc = -np.inf
    best_state = None
    cur_pat = 0
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for Xb, yb in train_loader:
            Xb = Xb.to(DEVICE).float()
            yb = yb.to(DEVICE).float()
            opt.zero_grad()
            logits = model(Xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * Xb.size(0)
        train_loss = total_loss / len(train_loader.dataset)
        val_auc, val_f1 = evaluate(val_loader, model)
        # print minimal progress
        print(f"  epoch {epoch}/{epochs} train_loss={train_loss:.4f} val_auc={val_auc:.4f} val_f1={val_f1:.4f}")
        if np.isfinite(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict()
            cur_pat = 0
        else:
            cur_pat += 1
        if cur_pat >= patience:
            break

    # load best and evaluate test
    if best_state is not None:
        model.load_state_dict(best_state)
    test_auc, test_f1 = evaluate(test_loader, model)
    return {'val_auc': float(best_val_auc), 'val_f1': float(val_f1), 'test_auc': float(test_auc), 'test_f1': float(test_f1), 'config': config}

def main():
    train, val, test = load_splits()
    feat_cols = [c for c in train.columns if c != 'target']
    X_train = train[feat_cols].to_numpy(dtype=np.float32)
    y_train = train['target'].to_numpy(dtype=np.float32)
    X_val = val[feat_cols].to_numpy(dtype=np.float32)
    y_val = val['target'].to_numpy(dtype=np.float32)
    X_test = test[feat_cols].to_numpy(dtype=np.float32)
    y_test = test['target'].to_numpy(dtype=np.float32)

    # small grid
    hidden_options = [[128,64], [256,128], [512,256]]
    dropout_options = [0.1, 0.2]
    lr_options = [1e-3, 5e-4]
    batch_options = [256, 512]

    grid = list(itertools.product(hidden_options, dropout_options, lr_options, batch_options))
    results = []
    start = time.time()
    for i, (hidden, dropout, lr, batch_size) in enumerate(grid, 1):
        print(f"Run {i}/{len(grid)}: hidden={hidden} dropout={dropout} lr={lr} batch={batch_size}")
        cfg = {'hidden_dims': hidden, 'dropout': dropout, 'lr': lr, 'batch_size': batch_size, 'epochs': 10, 'patience': 3}
        res = train_and_evaluate(cfg, X_train, y_train, X_val, y_val, X_test, y_test)
        res['run_index'] = i
        results.append(res)
        # persist intermediate results
        with open(MODEL_DIR / 'grid_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    duration = time.time() - start
    print(f"Grid search completed in {duration/60:.1f} minutes")
    # find best by val_auc
    best = max(results, key=lambda r: r['val_auc'] if np.isfinite(r['val_auc']) else -np.inf)
    print('Best config:', best['config'], 'val_auc=', best['val_auc'], 'test_auc=', best['test_auc'])
    # Save best model by re-training quickly to save state
    best_cfg = best['config']
    # retrain full on train+val? For simplicity, retrain on train+val combined then evaluate on test
    train_val = pd.concat([train, val])
    X_tv = train_val[feat_cols].to_numpy(dtype=np.float32)
    y_tv = train_val['target'].to_numpy(dtype=np.float32)
    model = MLP(input_dim=X_tv.shape[1], hidden_dims=tuple(best_cfg['hidden_dims']), dropout=best_cfg['dropout']).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=best_cfg['lr'], weight_decay=1e-5)
    crit = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(torch.tensor(X_tv), torch.tensor(y_tv)), batch_size=best_cfg['batch_size'], shuffle=True)
    for epoch in range(1, 11):
        model.train()
        for Xb, yb in loader:
            Xb = Xb.to(DEVICE).float(); yb = yb.to(DEVICE).float()
            opt.zero_grad(); logits = model(Xb); loss = crit(logits, yb); loss.backward(); opt.step()
    torch.save(model.state_dict(), MODEL_DIR / 'mlp_grid_best.pth')
    # evaluate
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=1024)
    test_auc, test_f1 = evaluate(test_loader, model)
    final = {'best_config': best_cfg, 'val_auc': best['val_auc'], 'val_f1': best['val_f1'], 'test_auc': test_auc, 'test_f1': test_f1}
    with open(MODEL_DIR / 'grid_best_metrics.json', 'w') as f:
        json.dump(final, f, indent=2)
    print('Saved grid best metrics to', MODEL_DIR / 'grid_best_metrics.json')

if __name__ == '__main__':
    main()
