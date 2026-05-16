import numpy as np

def kalman_filter(y, Z, T, H, Q):
    n = len(y)
    m = T.shape[0]
    
    # Inicializacao
    a = np.zeros((n + 1, m)) # Vetores de estado
    P = np.zeros((n + 1, m, m)) # Variancias de estado
    P[0] = np.eye(m) * 10 
    
    for t in range(n):
        # 1. Predicao (a_t|t-1, P_t|t-1)
        a_pred = T @ a[t]
        P_pred = T @ P[t] @ T.T + Q
        
        # 2. Atualizacao (Inovacao e Ganho)
        v = y[t] - Z @ a_pred  # Inovacao
        F = Z @ P_pred @ Z.T + H # Variancia da Inovacao
        K = P_pred @ Z.T * (1/F) # Ganho de Kalman
        
        a[t+1] = a_pred + K * v
        P[t+1] = (np.eye(m) - K @ Z) @ P_pred
        
    return a[1:], P[1:]
