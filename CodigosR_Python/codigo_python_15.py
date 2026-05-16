import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Gerando serie temporal do PIB sintetica
np.random.seed(42)
T = 300
trend = np.linspace(100, 200, T)
seasonal = 10*np.sin(np.linspace(0, 4*np.pi, T))
noise = np.random.normal(0, 5, T)
gdp = trend + seasonal + noise

# Pre-processamento: janela deslizante (window=12 meses)
scaler = MinMaxScaler()
gdp_scaled = scaler.fit_transform(gdp.reshape(-1,1))

def make_sequences(data, window=12):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

X_seq, y_seq = make_sequences(gdp_scaled, window=12)
X_t = torch.tensor(X_seq, dtype=torch.float32)
y_t = torch.tensor(y_seq, dtype=torch.float32)

# Reshape para (batch, seq_len, features)
X_t = X_t.unsqueeze(-1)

# Definir modelo LSTM
class LSTMForecast(nn.Module):
    def __init__(self, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden,
                           num_layers=layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMForecast()
optim = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(300):
    model.train()
    pred = model(X_t)
    loss = loss_fn(pred, y_t)
    optim.zero_grad()
    loss.backward()
    optim.step()
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.6f}')
