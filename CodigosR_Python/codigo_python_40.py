import numpy as np
from scipy.stats import genpareto
import matplotlib.pyplot as plt

# Simulando retornos com caudas pesadas
np.random.seed(99)
n = 5000
ret = np.random.standard_t(df=3, size=n) * 0.02

# Escolha do Limiar u (90o percentil das perdas)
perdas = -ret
u = np.quantile(perdas, 0.90)
exceedances = perdas[perdas > u] - u

# Estimacao da GPD via MLE
xi_hat, loc_hat, beta_hat = genpareto.fit(exceedances, floc=0)
print(f''Indice de Cauda xi: {xi_hat:.4f}'')
print(f''Escala beta: {beta_hat:.4f}'')

# VaR e ES implicitos pela GPD analitica
alpha = 0.99
n_exc = len(exceedances)
N = len(perdas)
VaR_evt = u + (beta_hat / xi_hat) * ((N*(1-alpha)/n_exc)**(-xi_hat) - 1)
ES_evt = (VaR_evt + beta_hat - xi_hat*u) / (1 - xi_hat)
print(f''VaR-EVT 99%: {VaR_evt:.4f}'')
print(f''ES-EVT 99%: {ES_evt:.4f}'')
