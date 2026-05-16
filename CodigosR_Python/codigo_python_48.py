import numpy as np
from scipy.optimize import minimize

def quantile_loss(beta, X, y, tau):
    errors = y - X @ beta
    return np.sum(np.where(errors >= 0, tau * errors, (tau - 1) * errors))

def estimar_regressao_quantilica(y, X, tau=0.5):
    # n obs, k regressores
    k = X.shape[1]
    initial_beta = np.zeros(k)
    
    # Minimizacao via algoritmo Simplex ou L-BFGS-B
    res = minimize(quantile_loss, initial_beta, args=(X, y, tau), method='Nelder-Mead')
    return res.x

# Comparacao Visual
# beta_median = estimar_regressao_quantilica(y, X, tau=0.5)
# beta_ols = np.linalg.inv(X.T @ X) @ X.T @ y
