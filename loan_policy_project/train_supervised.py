import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import joblib
import numpy as np
import pandas as pd
from data_utils import load_data, select_and_clean, encode_and_split

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)


def train(data_path, nrows=None, epochs=5, batch_size=256):
    df = load_data(data_path, nrows=nrows)
    proc = select_and_clean(df)
    X_train, X_test, y_train, y_test, scaler = encode_and_split(proc)

    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = MLP(X_train.shape[1])
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            opt.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        print(f'Epoch {epoch+1} loss = {total_loss/len(train_ds):.4f}')

    # Save model and scaler
    torch.save(model.state_dict(), 'supervised_mlp.pt')
    joblib.dump(scaler, 'scaler.joblib')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/accepted_2007_to_2018.csv')
    parser.add_argument('--nrows', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    train(args.data, nrows=args.nrows, epochs=args.epochs)
