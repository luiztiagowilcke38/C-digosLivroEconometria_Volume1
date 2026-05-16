import numpy as np
from scipy.optimize import minimize

def mnl_probabilities(beta, X, J):
    n, k = X.shape
    # beta_full = [beta2, beta3, ..., betaJ]
    beta_mat = beta.reshape(J-1, k)
    # Adicionando beta1 = 0 para referencia
    all_betas = np.vstack([np.zeros(k), beta_mat])
    
    # Utilitarios V = X * beta
    V = X @ all_betas.T
    expV = np.exp(V)
    sums = np.sum(expV, axis=1, keepdims=True)
    return expV / sums

def mnl_loglik(beta, X, y_indices, J):
    probs = mnl_probabilities(beta, X, J)
    # Selecionar a probabilidade da escolha realizada
    chosen_probs = probs[np.arange(len(y_indices)), y_indices]
    return -np.sum(np.log(chosen_probs + 1e-10))

# Dados sinteticos: 3 categorias, 2 regressores
n, J, k = 500, 3, 2
X = np.column_stack([np.ones(n), np.random.randn(n)])
true_betas = np.array([[0.5, 1.2], [-0.8, 2.5]]) # beta2 e beta3
V_true = np.hstack([np.zeros((n, 1)), X @ true_betas.T])
probs_true = np.exp(V_true) / np.sum(np.exp(V_true), axis=1, keepdims=True)
y = np.array([np.random.choice(J, p=p) for p in probs_true])

# Estimacao
initial_beta = np.zeros((J-1) * k)
res = minimize(mnl_loglik, initial_beta, args=(X, y, J), method='BFGS')
print("Betas estimados (referencia cat 0):")
print(res.x.reshape(J-1, k))
