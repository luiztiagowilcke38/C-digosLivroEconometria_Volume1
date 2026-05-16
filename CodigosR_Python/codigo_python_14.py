import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. Geracao de Variacao Nao Linear Pura de Fronteira (Painel Big Data)
np.random.seed(42)
torch.manual_seed(42)
N, p = 10000, 15
X_np = np.random.randn(N, p)
# Y e' gerado por um relâmpago Nao-Linear nao capturavel por MQO simples
Y_np = np.sin(X_np[:, 0] * 2) + np.log(np.abs(X_np[:, 1])+1) * 3 + X_np[:, 2]**2 + np.random.randn(N)

X_tensor = torch.tensor(X_np, dtype=torch.float32)
Y_tensor = torch.tensor(Y_np, dtype=torch.float32).view(-1, 1)

# 2. Especificando a Camada de Deep MLP Causal Substitutiva (Neural Net)
class EconometricsMLP(nn.Module):
    def __init__(self, in_features):
        super(EconometricsMLP, self).__init__()
        # 3 camadas intrinsecas nao-lineares
        self.fc1 = nn.Linear(in_features, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        return self.out(h)

# 3. Minimizacao MSE via Optimizer ADAM com Weight Decay (Penalty Ridge)
model = EconometricsMLP(in_features=p)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4) # Ridge Penalty

# 4. Loop Robusto para Treinamento Sub-Otimo
epochs = 500
for t in range(epochs):
    y_pred = model(X_tensor)      # Passagem Forward
    loss = loss_fn(y_pred, Y_tensor) # MQ Loss Function Equivalent
    
    optimizer.zero_grad()         # Zera buffers gradientes
    loss.backward()               # Back-Propagate por todas as Vieses Locais
    optimizer.step()              # Estocastico Gradient Step Parameter Update
    
    if t % 100 == 0:
        print(f''Epoca {t}: Otimizacao Local Loss Equivalente = {loss.item():.4f}'')
