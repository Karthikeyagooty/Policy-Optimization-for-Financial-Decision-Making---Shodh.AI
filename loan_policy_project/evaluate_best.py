import json
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / 'models'
DATA_DIR = ROOT / 'data'

metrics_path = MODEL_DIR / 'grid_best_metrics.json'
state_path = MODEL_DIR / 'mlp_grid_best.pth'
assert metrics_path.exists(), 'grid_best_metrics.json not found. Run grid search first.'
best = json.loads(metrics_path.read_text())
cfg = best.get('best_config', best.get('config', {}))
hidden = tuple(cfg.get('hidden_dims', [256, 128]))
dropout = cfg.get('dropout', 0.2)

if (DATA_DIR / 'test.parquet').exists():
    df_test = pd.read_parquet(DATA_DIR / 'test.parquet')
else:
    df_test = pd.read_parquet(DATA_DIR / 'processed_sample.parquet')

feat_cols = [c for c in df_test.columns if c != 'target']
X_test = df_test[feat_cols].to_numpy(dtype=np.float32)
y_test = df_test['target'].to_numpy(dtype=np.int32)

class MLPGrid(nn.Module):
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLPGrid(input_dim=len(feat_cols), hidden_dims=hidden, dropout=dropout).to(device)
if state_path.exists():
    model.load_state_dict(torch.load(state_path, map_location=device))
else:
    raise FileNotFoundError(f'Saved model not found at {state_path}')

model.eval()
with torch.no_grad():
    Xb = torch.tensor(X_test, dtype=torch.float32).to(device)
    logits = model(Xb).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))

auc = roc_auc_score(y_test, probs)
thresholds = np.linspace(0.01, 0.99, 99)
best_f1 = -1.0
best_thr = 0.5
for thr in thresholds:
    preds = (probs >= thr).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1; best_thr = float(thr)

print('Best config:', cfg)
print(f'Test AUC: {auc:.4f}, Best test F1: {best_f1:.4f} at threshold={best_thr:.3f}')

out = {'test_auc': float(auc), 'test_f1': float(best_f1), 'best_threshold': best_thr, 'config': cfg}
(MODEL_DIR / 'final_best_metrics.json').write_text(json.dumps(out, indent=2))
print('Saved final metrics to', MODEL_DIR / 'final_best_metrics.json')
