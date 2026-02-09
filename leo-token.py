import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# -------------------
# Load + sort
# -------------------
df = pd.read_csv('./leo-token.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# -------------------
# Train/Test split (time-based)
# -------------------
split_ratio = 0.8
split_idx = int(len(df) * split_ratio)

train_df = df.iloc[:split_idx].copy()
test_df  = df.iloc[split_idx:].copy()

# -------------------
# Scale (fit on train only!)
# -------------------
scalerX = MinMaxScaler()
scalery = MinMaxScaler()

X_train_raw = train_df[['total_volume', 'market_cap']].values
y_train_raw = train_df[['price']].values  # keep 2D

X_test_raw  = test_df[['total_volume', 'market_cap']].values
y_test_raw  = test_df[['price']].values

X_train = scalerX.fit_transform(X_train_raw)
y_train = scalery.fit_transform(y_train_raw)

X_test  = scalerX.transform(X_test_raw)
y_test  = scalery.transform(y_test_raw)

# Combine into one array for sequencing: [price, vol, cap]
train_data = np.concatenate([y_train, X_train], axis=1)
test_data  = np.concatenate([y_test,  X_test],  axis=1)

def create_sequences(data, seq_len=60):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len, 0])  # next price (scaled)
    return np.array(xs), np.array(ys).reshape(-1, 1)  # <-- make [N,1]

seq_len = 30
Xtr, ytr = create_sequences(train_data, seq_len)
Xte, yte = create_sequences(test_data, seq_len)

Xtr = torch.tensor(Xtr, dtype=torch.float32)
ytr = torch.tensor(ytr, dtype=torch.float32)
Xte = torch.tensor(Xte, dtype=torch.float32)
yte = torch.tensor(yte, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True)
test_loader  = DataLoader(TensorDataset(Xte, yte), batch_size=32, shuffle=False)

# -------------------
# Model
# -------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=16, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)


    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]          # last timestep hidden state
        out = self.fc(out)
        return out

model = LSTMModel(input_size=3, hidden_size=16)
loss_fn = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=1e-3)

# -------------------
# Train
# -------------------
epochs = 30
for ep in range(epochs):
    model.train()
    total = 0.0
    for xb, yb in train_loader:
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)    # shapes: [B,1] vs [B,1]
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"Epoch {ep+1}/{epochs}  Loss: {total/len(train_loader):.6f}")

# -------------------
# Evaluate on TEST
# -------------------
model.eval()
preds = []
trues = []
with torch.no_grad():
    for xb, yb in test_loader:
        preds.append(model(xb).cpu().numpy())
        trues.append(yb.cpu().numpy())

preds = np.vstack(preds)
trues = np.vstack(trues)

pred_price = scalery.inverse_transform(preds)
true_price = scalery.inverse_transform(trues)

mae = mean_absolute_error(true_price, pred_price)
rmse = np.sqrt(mean_squared_error(true_price, pred_price))
r2 = r2_score(true_price, pred_price)

print(f"TEST MAE: {mae:.3f}  RMSE: {rmse:.3f}  R2: {r2:.4f}")

# Align dates for plotting test predictions
test_dates = test_df['date'].iloc[seq_len:].reset_index(drop=True)

plt.figure(figsize=(18,4))
plt.plot(test_dates, true_price, label="Actual")
plt.plot(test_dates, pred_price, label="Predicted")
plt.title("Test Set: Actual vs Predicted")
plt.grid(True)
plt.legend()
plt.show()


future_days = 7

# Build scaled full data using scalers fit on train
y_all_scaled = scalery.transform(df[['price']].values)  # (N,1)
X_all_scaled = scalerX.transform(df[['total_volume', 'market_cap']].values)  # (N,2)

scaled_all = np.concatenate([y_all_scaled, X_all_scaled], axis=1)  # (N,3)

# last 60 days of scaled [price, vol, cap]
last_seq = scaled_all[-seq_len:]  # (60,3)

pred_scaled = []
model.eval()

with torch.no_grad():
    current_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0)  # (1,60,3)

    for _ in range(future_days):
        next_price_scaled = model(current_seq)   # (1,1)
        pred_scaled.append(next_price_scaled.item())

        # keep last known vol & cap (future unknown)
        last_vol = current_seq[0, -1, 1].item()
        last_cap = current_seq[0, -1, 2].item()

        # next row: [pred_price, last_vol, last_cap]
        next_row = torch.tensor([[next_price_scaled.item(), last_vol, last_cap]], dtype=torch.float32)  # (1,3)

        # slide the window: drop oldest, append new
        current_seq = torch.cat([current_seq[:, 1:, :], next_row.unsqueeze(0)], dim=1)  # (1,60,3)

# inverse transform back to real prices
pred_prices = scalery.inverse_transform(np.array(pred_scaled).reshape(-1, 1))

last_date = df['date'].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

print("Next 7 days forecast:")
for d, p in zip(future_dates, pred_prices.flatten()):
    print(d.date(), float(p))

plt.figure(figsize=(16,6))
plt.plot(df['date'], df['price'], label='Historical')
plt.plot(future_dates, pred_prices, label='Next 7 days forecast')
plt.title('Bitcoin Price Forecast (Next 7 Days)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

