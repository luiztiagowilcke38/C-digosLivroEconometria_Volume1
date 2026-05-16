import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class EconometricLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(EconometricLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Arquitetura Principal LSTM
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim,
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout)
                            
        # Camada de Regressao Densa Plena Fully-Connected
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x.shape := (batch_size, sequence_length, input_dim)
        # H_0 e C_0 inicializam como zero implicito por padrao
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Pegamos o output do ultimo Time Step
        last_time_step = lstm_out[:, -1, :] 
        
        # Passamos na decaida densa
        out = self.fc(last_time_step)
        return out

def treinar_lstm_gpu(X_train, y_train, epochs=100, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Tensor Conversion para Gradientes Acelerados
    tensor_X = torch.tensor(X_train, dtype=torch.float32).to(device)
    tensor_y = torch.tensor(y_train, dtype=torch.float32).to(device)
    
    loader = DataLoader(TensorDataset(tensor_X, tensor_y), batch_size=batch_size, shuffle=True)
    
    model = EconometricLSTM(input_dim=X_train.shape[2], hidden_dim=128, num_layers=3, output_dim=1).to(device)
    criterion = nn.HuberLoss() # Mais robusto que MSE a outliers (Huber)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad() # Zerar matrizes hessianas em cache
            preds = model(batch_x)
            loss = criterion(preds.squeeze(), batch_y)
            loss.backward() # Backpropagation Algoritmico Automatico
            
            # Gradient clipping para expliding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
    return model

# modelo_treinado = treinar_lstm_gpu(X_sequencial, Y_volatilidade)
