import numpy as np
import scipy.optimize as opt
from scipy.special import gammaln

def negbin_log_likelihood(params, Y, X):
    ''''''
    Parametros: beta, alpha (onde alpha = variancia / mean - 1)
    ''''''
    k = X.shape[1]
    beta = params[:k]
    alpha = np.exp(params[k]) # reparametrizado para evitar alpha <= 0
    
    mu = np.exp(X @ beta)
    
    # Derivacao de NegBin-2 MLE function
    term1 = gammaln(Y + 1/alpha) - gammaln(Y + 1) - gammaln(1/alpha)
    term2 = Y * np.log(alpha * mu)
    term3 = - (Y + 1/alpha) * np.log(1 + alpha * mu)
    
    ll = np.sum(term1 + term2 + term3)
    return -ll # Minimizador scipy requer -LL

def negbin_jacobian(params, Y, X):
    k = X.shape[1]
    beta = params[:k]
    alpha = np.exp(params[k])
    
    mu = np.exp(X @ beta)
    
    # Derivadas com relacao a beta: d(LL)/d(beta_j)
    dLL_dbeta = X.T @ ((Y - mu) / (1 + alpha * mu))
    
    # Derivadas com relacao a ln(alpha) via regra da cadeia
    from scipy.special import digamma
    dLL_dalpha_raw = (
        - (alpha**-2) * (digamma(Y + 1/alpha) - digamma(1/alpha)) 
        + Y/alpha 
        + (alpha**-2) * np.log(1 + alpha*mu) 
        - (Y + 1/alpha)*(mu) / (1 + alpha*mu)
    )
    dLL_dlogalpha = np.sum(dLL_dalpha_raw) * alpha
    
    grad = np.append(dLL_dbeta, dLL_dlogalpha)
    return -grad # gradiente da negativa-verossimilhanca

def estimar_negbin(Y, X):
    k = X.shape[1]
    # Chute Minimos Quadrados Iterativos (IRLS) - Poisson Approximation
    beta_init = np.linalg.lstsq(X, np.log(Y + 0.1), rcond=None)[0]
    initial_params = np.append(beta_init, 0.0) # alpha = exp(0) = 1
    
    # Otimizacao via L-BFGS-B
    res = opt.minimize(
        fun=negbin_log_likelihood,
        x0=initial_params,
        args=(Y, X),
        method='L-BFGS-B',
        jac=negbin_jacobian,
        options={'disp': True, 'maxiter': 500, 'ftol': 1e-11}
    )
    
    # O Padrao Ouro em Inferencia
    hess_inv_approx = res.hess_inv.todense() if hasattr(res.hess_inv, 'todense') else res.hess_inv
    std_errors = np.sqrt(np.diag(hess_inv_approx))
    return res.x, std_errors

warnings = False 
# beta_hat, se = estimar_negbin(y_count, X_vars)
