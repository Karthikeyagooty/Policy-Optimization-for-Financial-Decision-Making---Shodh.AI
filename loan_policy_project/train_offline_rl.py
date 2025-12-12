"""
Train an offline RL agent (CQL) using d3rlpy on the built RL dataset.

Workflow:
- Load `data/rl_dataset.npz` (or build it if missing).
- Create a d3rlpy MDPDataset and train CQL.
- Evaluate learned policy by computing expected reward on a held-out test split.
"""
from pathlib import Path
import numpy as np
import json
import d3rlpy
from d3rlpy.algos import CQL
from d3rlpy.dataset import MDPDataset
import warnings

ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data'
MODEL_DIR = ROOT / 'models'
MODEL_DIR.mkdir(exist_ok=True)

def build_if_missing():
    if not (DATA_DIR / 'rl_dataset.npz').exists():
        print('RL dataset missing; building...')
        import build_rl_dataset
        build_rl_dataset.main()

def load_npz():
    arr = np.load(DATA_DIR / 'rl_dataset.npz')
    obs = arr['observations']
    acts = arr['actions'].squeeze(-1)
    rews = arr['rewards'].squeeze(-1)
    terms = arr['terminals'].squeeze(-1).astype(bool)
    return obs, acts, rews, terms

def make_dataset(obs, acts, rews, terms):
    # d3rlpy accepts array-style MDPDataset for single-step episodes.
    # Ensure shapes: obs (n, dim), acts (n,), rews (n,), terms (n,)
    acts = np.asarray(acts).squeeze()
    rews = np.asarray(rews).squeeze()
    terms = np.asarray(terms).squeeze().astype(bool)
    obs = np.asarray(obs)
    return MDPDataset(observations=obs, actions=acts, rewards=rews, terminals=terms)

def expected_reward_on_test(policy, X_test, y_test, loan_amnt, int_rate):
    # policy.predict expects 2D array
    acts = policy.predict(X_test)
    # if discrete actions, might return shape (n,); ensure ints
    acts = np.asarray(acts).astype(int).squeeze()
    rewards = np.zeros(len(acts), dtype=float)
    fully_paid = (y_test == 0)
    approve_mask = acts == 1
    rewards[approve_mask & fully_paid] = int_rate[approve_mask & fully_paid]
    rewards[approve_mask & (~fully_paid)] = -loan_amnt[approve_mask & (~fully_paid)]
    return float(rewards.mean())

def main():
    build_if_missing()
    obs, acts, rews, terms = load_npz()
    # split into train/test for policy evaluation
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(obs))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=acts)
    obs_train = obs[train_idx]
    acts_train = acts[train_idx]
    rews_train = rews[train_idx]
    terms_train = terms[train_idx]
    obs_test = obs[test_idx]
    acts_test = acts[test_idx]
    rews_test = rews[test_idx]

    dataset = make_dataset(obs_train, acts_train, rews_train, terms_train)

    # configure and train Discrete CQL
    print('Training Discrete CQL...')
    from d3rlpy.algos import DiscreteCQL, DiscreteCQLConfig
    cfg = DiscreteCQLConfig()
    algo = DiscreteCQL(cfg, device='cpu', enable_ddp=False)
    # fit (short run to keep runtime reasonable)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # run for a modest number of gradient steps
        algo.fit(dataset, n_steps=10000, n_steps_per_epoch=2000)

    # save algo
    algo.save_model(str(MODEL_DIR / 'discrete_cql_model'))

    # Evaluate policy by expected reward on test observations
    # Load test meta for loan_amnt/int_rate
    import pandas as pd
    df = pd.read_parquet(DATA_DIR / 'processed_sample.parquet') if (DATA_DIR / 'processed_sample.parquet').exists() else None
    if df is None:
        print('processed_sample.parquet not found for evaluation metadata; using obs_test only')
        # fallback: cannot compute monetary reward without loan_amnt/int_rate; report action distribution instead
        policy = algo
        acts_pred = policy.predict(obs_test)
        avg_action = float(np.mean(acts_pred))
        out = {'avg_action_test': avg_action}
        (MODEL_DIR / 'offline_rl_metrics.json').write_text(json.dumps(out, indent=2))
        print('Saved offline_rl_metrics.json (action distribution fallback)')
        return

    # we need to index the original dataframe rows used in obs_test; simpler: reload full df and compute test mask by split
    # Rebuild the same split indices used in build_rl_dataset
    full_df = pd.concat([pd.read_parquet(DATA_DIR / 'train.parquet')] + ([pd.read_parquet(DATA_DIR / 'val.parquet')] if (DATA_DIR / 'val.parquet').exists() else []), ignore_index=True)
    # If sizes mismatch, just use processed_sample as source
    if len(full_df) != len(obs):
        full_df = pd.read_parquet(DATA_DIR / 'processed_sample.parquet')

    loan_amnt = full_df['loan_amnt'].to_numpy(dtype=np.float32)[test_idx]
    int_rate = full_df['int_rate'].to_numpy(dtype=np.float32)[test_idx]
    y_test = full_df['target'].to_numpy(dtype=int)[test_idx]

    policy = algo
    # d3rlpy policy object supports predict(observations)
    val = expected_reward_on_test(policy, obs_test, y_test, loan_amnt, int_rate)
    out = {'expected_reward_test': float(val)}
    (MODEL_DIR / 'offline_rl_metrics.json').write_text(json.dumps(out, indent=2))
    print('Saved offline_rl_metrics.json with expected_reward_test=', val)

if __name__ == '__main__':
    main()
