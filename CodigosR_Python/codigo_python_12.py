import numpy as np
import statsmodels.api as sm
from joblib import Parallel, delayed
import time

def dgp_dynamic_panel(N, T, rho_true, beta_true, gamma, var_eps, var_v):
    ''''''
    Gera DGP para: y_{it} = rho * y_{i,t-1} + beta * x_{it} + alpha_i + eps_{it}
    ''''''
    alpha = np.random.normal(0, 1, N)
    eta = np.random.normal(0, 1, N)
    
    Y = np.zeros((N, T))
    X = np.zeros((N, T))
    
    # Condicoes iniciais estacionarias se |rho| < 1
    Y[:, 0] = alpha / (1 - rho_true) + np.random.normal(0, np.sqrt(var_eps/(1-rho_true**2)), N)
    X[:, 0] = eta / (1 - gamma) + np.random.normal(0, np.sqrt(var_v/(1-gamma**2)), N)
    
    for t in range(1, T):
        X[:, t] = gamma * X[:, t-1] + eta + np.random.normal(0, np.sqrt(var_v), N)
        Y[:, t] = rho_true * Y[:, t-1] + beta_true * X[:, t] + alpha + np.random.normal(0, np.sqrt(var_eps), N)
        
    return Y, X

def estimar_fe_within(Y, X):
    N, T = Y.shape
    # Transforma para desviacoes da media temporal (Within transformation)
    Y_mean = np.mean(Y, axis=1, keepdims=True)
    X_mean = np.mean(X, axis=1, keepdims=True)
    
    y_tilde = (Y[:, 1:] - Y_mean).flatten()
    
    # Regressores: y_{t-1} e X_{t}
    y_lag_tilde = (Y[:, :-1] - Y_mean).flatten()
    x_tilde = (X[:, 1:] - X_mean).flatten()
    
    Regressors = np.column_stack((y_lag_tilde, x_tilde))
    
    # OLS na equacao transformada
    try:
        model = sm.OLS(y_tilde, Regressors).fit()
        return model.params
    except:
        return np.array([np.nan, np.nan])

def monte_carlo_iteration(N, T, rho, beta):
    np.random.seed() # Re-seed the worker
    Y, X = dgp_dynamic_panel(N, T, rho, beta, gamma=0.5, var_eps=1.0, var_v=1.0)
    return estimar_fe_within(Y, X)

def vies_nickell_paralelo(N=100, T=10, R=2000, rho=0.8, beta=1.0, n_jobs=-1):
    start = time.time()
    resultados = Parallel(n_jobs=n_jobs)(
        delayed(monte_carlo_iteration)(N, T, rho, beta) for _ in range(R)
    )
    resultados = np.array(resultados)
    resultados = resultados[~np.isnan(resultados).any(axis=1)] # drop NAs
    
    rho_estimado = np.mean(resultados[:, 0])
    beta_estimado = np.mean(resultados[:, 1])
    
    vies_rho = rho_estimado - rho
    print(f''Monte Carlo ({R} runs, N={N}, T={T}) | Tempo: {time.time()-start:.2f}s'')
    print(f''Beta True: {beta} | Beta Hat: {beta_estimado:.4f}'')
    print(f''Rho True:  {rho} | Rho Hat:  {rho_estimado:.4f} (Vies de {vies_rho:.4f})'')
    
    return vies_rho

# Execucao do Experimento
# bias = vies_nickell_paralelo(N=200, T=5, R=5000)
