import numpy as np
from scipy import stats

def teste_f_customizado(y, X, R, r_vec):
    # n obs, k regressores, q restricoes
    n, k = X.shape
    q = R.shape[0]
    
    # 1. Estimacao MQO
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ y
    residuos = y - X @ beta_hat
    s2 = (residuos.T @ residuos) / (n - k)
    
    # 2. Estatistica F
    diff = R @ beta_hat - r_vec
    middle = np.linalg.inv(R @ XtX_inv @ R.T)
    f_stat = (diff.T @ middle @ diff) / (q * s2)
    
    p_value = 1 - stats.f.cdf(f_stat, q, n-k)
    return f_stat, p_value

# Exemplo: Testar se b1 = b2
# R = np.array([[0, 1, -1]]) # Assumindo intercepto, b1 e b2
# r_vec = np.array([0])
