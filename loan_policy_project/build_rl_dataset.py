"""
Build an offline RL dataset from processed train/val splits.

Approach:
- Load `train.parquet` and `val.parquet` (or `processed_sample.parquet` fallback).
- Train a simple logistic classifier to estimate approval probability (behavior policy).
- Sample actions ~ Bernoulli(p_approve) to create logged actions (so dataset contains both approve/deny).
- Compute rewards:
    - If action == 0: reward = 0
    - If action == 1 and target == 0 (fully paid): reward = +int_rate
    - If action == 1 and target == 1 (default): reward = -loan_amnt
- Save dataset arrays to `data/rl_dataset.npz` and also return a d3rlpy MDPDataset when run directly.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import json

ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data'
OUT_PATH = DATA_DIR / 'rl_dataset.npz'

def load_data():
    if (DATA_DIR / 'train.parquet').exists() and (DATA_DIR / 'val.parquet').exists():
        train = pd.read_parquet(DATA_DIR / 'train.parquet')
        val = pd.read_parquet(DATA_DIR / 'val.parquet')
        df = pd.concat([train, val], ignore_index=True)
    else:
        df = pd.read_parquet(DATA_DIR / 'processed_sample.parquet')
    return df

def build_dataset(df, random_state=42, deny_smooth=0.02):
    # features: everything except target
    feat_cols = [c for c in df.columns if c != 'target']
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df['target'].to_numpy().astype(int)
    # Use loan_amnt and int_rate for reward computation; ensure they exist
    assert 'loan_amnt' in df.columns and 'int_rate' in df.columns, 'Required columns missing'
    loan_amnt = df['loan_amnt'].to_numpy(dtype=np.float32)
    int_rate = df['int_rate'].to_numpy(dtype=np.float32)

    # Train small classifier to estimate repay probability (1 - default prob)
    # Use a random 80/20 split for training the behavior model
    X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    clf = LogisticRegression(max_iter=200)
    try:
        clf.fit(X_train, y_train)
        p_default = clf.predict_proba(X)[:, 1]
    except Exception:
        # fallback: use empirical default rate
        p_default = np.full(len(y), y.mean())

    p_approve = 1.0 - p_default
    # small smoothing to avoid p=0/1
    p_approve = p_approve * (1 - deny_smooth) + 0.5 * deny_smooth

    rng = np.random.default_rng(random_state)
    actions = rng.binomial(1, p_approve).astype(np.int32)

    # Rewards per spec
    rewards = np.zeros(len(actions), dtype=np.float32)
    approve_mask = actions == 1
    # target==0 means fully paid (profit)
    fully_paid = (y == 0)
    rewards[approve_mask & fully_paid] = int_rate[approve_mask & fully_paid]
    rewards[approve_mask & (~fully_paid)] = -loan_amnt[approve_mask & (~fully_paid)]

    # episodes: single-step MDP; terminals True
    terminals = np.ones(len(actions), dtype=bool)

    return {
        'observations': X,
        'actions': actions[:, None],
        'rewards': rewards[:, None],
        'terminals': terminals[:, None],
        'feat_cols': feat_cols
    }

def save_npz(data_dict, out_path=OUT_PATH):
    np.savez_compressed(out_path, observations=data_dict['observations'], actions=data_dict['actions'], rewards=data_dict['rewards'], terminals=data_dict['terminals'])
    # also write metadata
    meta = {'feat_cols': data_dict['feat_cols']}
    (DATA_DIR / 'rl_dataset_meta.json').write_text(json.dumps(meta))
    print('Saved RL dataset to', out_path)

def main():
    df = load_data()
    ds = build_dataset(df)
    save_npz(ds)

if __name__ == '__main__':
    main()
