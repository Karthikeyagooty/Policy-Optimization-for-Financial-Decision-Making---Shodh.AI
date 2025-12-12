import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURES = [
    'loan_amnt', 'term', 'int_rate', 'annual_inc', 'dti', 'emp_length',
    'home_ownership', 'purpose', 'addr_state', 'delinq_2yrs', 'inq_last_6mths'
]


def load_data(path, nrows=None):
    """Load the LendingClub CSV file."""
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    return df


def select_and_clean(df, features=DEFAULT_FEATURES):
    """Select features, basic cleaning, and convert target to binary.

    Returns processed DataFrame with `target` column (0: fully paid, 1: defaulted).
    """
    # copy
    d = df.copy()

    # target mapping: 'Fully Paid' -> 0, 'Charged Off' -> 1, others dropped
    def map_status(x):
        if pd.isna(x):
            return np.nan
        if 'Fully Paid' in x:
            return 0
        if 'Charged Off' in x:
            return 1
        return np.nan

    d['target'] = d['loan_status'].map(map_status)
    d = d.dropna(subset=['target'])

    # simplify 'term' (e.g., ' 36 months')
    if 'term' in d.columns:
        d['term'] = d['term'].astype(str).str.extract(r"(\d+)").astype(float)

    # emp_length: convert to years
    if 'emp_length' in d.columns:
        d['emp_length'] = d['emp_length'].astype(str).str.extract(r"(\d+)")
        d['emp_length'] = pd.to_numeric(d['emp_length'], errors='coerce')

    # interest rate: remove %
    if 'int_rate' in d.columns:
        d['int_rate'] = d['int_rate'].astype(str).str.rstrip('%').astype(float)

    # fill numeric NaNs with median
    num_cols = d.select_dtypes(include=['number']).columns.tolist()
    for c in num_cols:
        d[c] = d[c].fillna(d[c].median())

    # categorical fill
    cat_cols = d.select_dtypes(include=['object']).columns.tolist()
    for c in cat_cols:
        d[c] = d[c].fillna('missing')

    # keep subset
    keep = [c for c in features if c in d.columns]
    keep += ['target']
    proc = d[keep].copy()

    return proc


def encode_and_split(df, cat_cols=None, test_size=0.2, random_state=42):
    """Encode categorical variables (one-hot for small cardinality), scale numerics, and split."""
    X = df.drop(columns=['target']).copy()
    y = df['target'].astype(int).copy()

    if cat_cols is None:
        # heuristic: object cols
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    scaler = StandardScaler()
    num_cols = X_enc.select_dtypes(include=['number']).columns.tolist()
    X_enc[num_cols] = scaler.fit_transform(X_enc[num_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler


def build_rl_dataset(df, features=None):
    """Given the processed df (with `target`), build an offline RL dataset.

    - State: feature vector
    - Action: if we have an observed decision column `is_approved`, use it; otherwise assume all were approved (dataset contains only accepted loans) and set action=1
    - Reward: per spec: Deny -> 0, Approve & paid -> +loan_amnt*int_rate, Approve & default -> -loan_amnt

    Returns: DataFrame with columns: state (array), action, reward
    """
    d = df.copy()
    if features is None:
        features = [c for c in d.columns if c != 'target']

    # observed approvals: LendingClub accepted loans - assume action=1 for all
    d['action'] = 1

    # compute reward
    if 'loan_amnt' in d.columns and 'int_rate' in d.columns:
        d['reward'] = d.apply(lambda r: (r['loan_amnt'] * r['int_rate'] / 100.0) if r['target'] == 0 and r['action'] == 1 else (-r['loan_amnt'] if r['target'] == 1 and r['action'] == 1 else 0), axis=1)
    else:
        d['reward'] = d['action'] * (1 - d['target'])

    # state as numpy array
    states = d[features].values
    actions = d['action'].astype(int).values
    rewards = d['reward'].astype(float).values

    return states, actions, rewards
