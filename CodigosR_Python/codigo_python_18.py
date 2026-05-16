import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

np.random.seed(0)
n = 2000
# Simulando GARCH(1,1) verdadeiro: omega=0.01, alpha=0.1, beta=0.85
omega_t, alpha_t, beta_t = 0.01, 0.10, 0.85
sigma2 = np.zeros(n)
eps    = np.zeros(n)
sigma2[0] = omega_t / (1 - alpha_t - beta_t)  # Variancia incondicional
for t in range(1, n):
    sigma2[t] = omega_t + alpha_t * eps[t-1]**2 + beta_t * sigma2[t-1]
    eps[t] = np.random.normal(0, np.sqrt(sigma2[t]))
ret = eps  # Serie de retornos observada

def garch_loglik(params, r):
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha+beta >= 1:
        return 1e10
    T = len(r)
    sig2 = np.empty(T)
    sig2[0] = np.var(r)
    for t in range(1, T):
        sig2[t] = omega + alpha * r[t-1]**2 + beta * sig2[t-1]
    # Log-Verossimilhança Normal Condicional (negativa para minimizacao)
    ll = -0.5 * np.sum(np.log(2*np.pi) + np.log(sig2) + r**2 / sig2)
    return -ll  # Minimizar

# Estimar via Nelder-Mead partindo de chute inicial plausivel
x0 = [0.02, 0.08, 0.88]
result = minimize(garch_loglik, x0, args=(ret,),
                  method='Nelder-Mead',
                  options={'maxiter': 5000, 'xatol': 1e-7})
omega_h, alpha_h, beta_h = result.x
print(f''omega: {omega_h:.5f} (verdadeiro {omega_t})'')
print(f''alpha: {alpha_h:.5f} (verdadeiro {alpha_t})'')
print(f''beta:  {beta_h:.5f} (verdadeiro {beta_t})'')
print(f''Persistencia: {alpha_h+beta_h:.5f}'')
