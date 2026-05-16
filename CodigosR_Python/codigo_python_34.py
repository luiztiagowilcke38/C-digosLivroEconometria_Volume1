import numpy as np
from scipy.stats import invgamma, multivariate_normal

def gibbs_linear_regression(y, X, draws=5000):
    n, k = X.shape
    # Prioris nao-informativas
    m0 = np.zeros(k)
    C0 = np.eye(k) * 1e6
    a0, b0 = 0.01, 0.01 # Priori para sigma^2 (Inv-Gamma)
    
    # Inicializacao
    beta = np.zeros(k)
    sigma2 = 1.0
    
    trace_beta = []
    trace_sigma2 = []
    
    XtX = X.T @ X
    XtY = X.T @ y
    
    for s in range(draws):
        # 1. Update Beta | sigma2, y
        V_n = inv(inv(C0) + XtX / sigma2)
        m_n = V_n @ (inv(C0) @ m0 + XtY / sigma2)
        beta = multivariate_normal.rvs(mean=m_n, cov=V_n)
        
        # 2. Update sigma2 | beta, y
        an = a0 + n / 2
        bn = b0 + 0.5 * np.sum((y - X @ beta)**2)
        sigma2 = invgamma.rvs(an, scale=bn)
        
        trace_beta.append(beta)
        trace_sigma2.append(sigma2)
        
    return np.array(trace_beta), np.array(trace_sigma2)

# O uso de amostras de Gibbs permite Inferencia Exata sem calculo de integrais.
