# Loan Policy Project

This repository contains code and notebooks for the Loan Policy assignment: EDA, a supervised DL model (MLP), and an offline RL agent (CQL).

Contents
- `1_EDA.ipynb` — Exploratory Data Analysis and preprocessing. Saves processed splits to `data/`.
- `2_supervised.ipynb` — Supervised MLP notebook; includes evaluation cell to load best model.
- `train_mlp.py` / `train_mlp_grid.py` — Scripts to train MLP (single run and grid search).
- `evaluate_best.py` — Load grid best model and compute final AUC/F1 (threshold sweep).
- `build_rl_dataset.py` — Build an offline RL dataset (observations, actions, rewards) from processed data.
- `train_offline_rl.py` — Train a Discrete CQL agent (d3rlpy) on the offline dataset and evaluate expected reward.
- `models/` — Saved model artifacts and metrics JSONs.
- `data/` — Processed data splits and RL dataset `.npz`.
- `reports/` — Generated final report (DOCX/PDF).

Reproducible Environment
1. Create and activate a Python virtual environment (project uses `.venv`):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
# If you plan to generate the DOCX report automatically:
pip install python-docx
```

Reproduce the experiments (quick commands)

1. Run the supervised grid search (creates `models/grid_best_metrics.json` and `models/mlp_grid_best.pth`):

```bash
.venv/bin/python train_mlp_grid.py
```

2. Evaluate the best supervised model and save final metrics:

```bash
.venv/bin/python evaluate_best.py
```

3. Build RL dataset and train CQL (offline RL):

```bash
.venv/bin/python build_rl_dataset.py
.venv/bin/python train_offline_rl.py
```

4. Build the final report (DOCX):

```bash
.venv/bin/python build_report.py
# This creates `reports/final_report.docx` (and `reports/final_report.md`).
```

Notes
- All generated artifacts (models, metrics, logs) are saved in `models/`, `data/`, and `d3rlpy_logs/` respectively.
- The report builder reads metrics from `models/final_best_metrics.json`, `models/grid_best_metrics.json`, and `models/offline_rl_metrics.json`. Ensure these files exist before running `build_report.py`.

If you want a PDF instead of DOCX, open the generated DOCX and export to PDF using Word or LibreOffice.
**Loan Policy Optimization — Project Scaffold**

- **Overview**: This repository contains notebooks and helper code to implement the steps from the assignment: EDA, a supervised deep learning model, and an offline RL agent for loan approval decisions.

- **Dataset**: `accepted_2007_to_2018.csv` from the LendingClub dataset (Kaggle). Place the file at `./data/accepted_2007_to_2018.csv` or use the Kaggle CLI to download it.

Quick setup:

1. Create and activate a Python environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Download dataset (Kaggle API):

- Place your `kaggle.json` in `~/.kaggle/kaggle.json` (or run `kaggle config set`), then:

```bash
kaggle datasets download -d 'wordsforthewise/lending-club' -f accepted_2007_to_2018.csv -p ./data --unzip
```

3. Notebooks:
- `1_EDA.ipynb` — EDA and preprocessing
- `2_supervised.ipynb` — Supervised MLP training and evaluation
- `3_offline_rl.ipynb` — Construct offline RL dataset and train BC / template for CQL using `d3rlpy`

4. Files of interest:
- `data_utils.py` — helper functions for loading and preprocessing
- `train_supervised.py` — (optional) script to run training from CLI

Notes and next steps:
- This scaffold assumes the dataset will be downloaded into `./data/`.
- Running the RL notebook requires `d3rlpy` and a CPU/GPU-enabled PyTorch installation; see `d3rlpy` docs for installation help.

If you want, I can now:
- implement core preprocessing in `data_utils.py` and populate notebooks with runnable code cells, or
- download the dataset here (I need your Kaggle API token to do that), then run and train models and return results.

Which do you prefer?