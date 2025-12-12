import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

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
        print('Loaded splits from data dir')
    else:
        sample = DATA_DIR / 'processed_sample.parquet'
        assert sample.exists(), 'No processed data found. Run 1_EDA.ipynb first.'
        df = pd.read_parquet(sample)
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
        train, val = train_test_split(train, test_size=0.125, random_state=42, stratify=train['target'])
        print('Created splits from processed_sample')
    return train, val, test

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
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

def main():
    train, val, test = load_splits()
    feature_cols = [c for c in train.columns if c != 'target']
    X_train = train[feature_cols].to_numpy(dtype=np.float32)
    y_train = train['target'].to_numpy(dtype=np.float32)
    X_val = val[feature_cols].to_numpy(dtype=np.float32)
    y_val = val['target'].to_numpy(dtype=np.float32)
    X_test = test[feature_cols].to_numpy(dtype=np.float32)
    y_test = test['target'].to_numpy(dtype=np.float32)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=512, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=1024)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=1024)

    model = MLP(input_dim=X_train.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    patience = 5
    pat = 0
    for epoch in range(1, 21):
        model.train()
        total_loss = 0.0
        for Xb, yb in train_loader:
            Xb = Xb.to(DEVICE).float()
            yb = yb.to(DEVICE).float()
            opt.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * Xb.size(0)
        train_loss = total_loss / (len(train_loader.dataset))
        val_auc, val_f1 = evaluate(val_loader, model)
        print(f'Epoch {epoch}: train_loss={train_loss:.4f} val_auc={val_auc:.4f} val_f1={val_f1:.4f}')
        if np.isfinite(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), MODEL_DIR / 'mlp_best.pth')
            pat = 0
        else:
            pat += 1
        if pat >= patience:
            print('Early stopping')
            break

    # Evaluate
    model.load_state_dict(torch.load(MODEL_DIR / 'mlp_best.pth', map_location=DEVICE))
    test_auc, test_f1 = evaluate(test_loader, model)
    print('Test AUC: {:.4f}, Test F1: {:.4f}'.format(test_auc, test_f1))
    with open(MODEL_DIR / 'metrics.json', 'w') as f:
        json.dump({'test_auc': float(test_auc), 'test_f1': float(test_f1)}, f, indent=2)
    print('Saved metrics to', MODEL_DIR / 'metrics.json')

if __name__ == '__main__':
    main()
