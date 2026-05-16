import numpy as np
from scipy import integrate
from scipy.stats import norm, poisson

# =========================================================
# PARTE A: Verificacao da Equivalencia para Vars Continuas
# =========================================================
# E[X] onde X ~ N(mu, sigma^2). Verdadeiro = mu = 3.0
mu, sigma = 3.0, 1.5

# (1) Analitico (formula fechada)
print(f''E[X] analitico (Normal):    {mu:.4f}'')

# (2) Integral de Riemann numerica (scipy quad)
f_x = lambda x: x * norm.pdf(x, mu, sigma)
E_riemann, err = integrate.quad(f_x, -np.inf, np.inf)
print(f''E[X] Riemann (scipy.quad):  {E_riemann:.4f} (erro est.: {err:.2e})'')

# (3) Integral de Lebesgue numerica (aproximacao por funcoes simples)
# Lebesgue: particiona o CONTRADOMINIO em fatias e soma x * P(X in fatia)
N_fatias = 10000
x_vals = np.linspace(mu - 6*sigma, mu + 6*sigma, N_fatias)
delta_x = x_vals[1] - x_vals[0]
# Aproximacao por funcao simples: soma de x_k * P(X em [x_k, x_k+delta])
probs_fatia = norm.pdf(x_vals, mu, sigma) * delta_x
E_lebesgue  = np.sum(x_vals * probs_fatia)
print(f''E[X] Lebesgue (fatias):     {E_lebesgue:.4f}'')

# (4) Monte Carlo (Lei dos Grandes Numeros = convergencia de E Lebesgue)
amostras = np.random.normal(mu, sigma, 1_000_000)
E_mc = np.mean(amostras)
print(f''E[X] Monte Carlo (N=1M):    {E_mc:.4f}'')

# =========================================================
# PARTE B: Variavel Discreta - Medida de Contagem (Poisson)
# =========================================================
lam = 4.5  # Parametro Poisson
ks = np.arange(0, 30)
E_discrete = np.sum(ks * poisson.pmf(ks, lam))
print(f''\nE[X] Poisson(4.5) analitico:   {lam:.4f}'')
print(f''E[X] Poisson soma de Lebesgue: {E_discrete:.4f}'')

# =========================================================
# PARTE C: Variavel MISTA - Modelo Tobit (Y = max(0, Y*))
# =========================================================
# Y* ~ N(X*beta, sigma^2); Y = max(0, Y*)
# E[Y] = Phi(X*beta/sigma) * (X*beta) + sigma * phi(X*beta/sigma)
# onde Phi=CDF Normal, phi=PDF Normal

np.random.seed(42)
n = 100_000
X_vec  = np.random.normal(0, 1, n)
beta_t = 2.0
sig_t  = 1.5
Y_star = X_vec * beta_t + np.random.normal(0, sig_t, n)
Y_obs  = np.maximum(0, Y_star)  # Tobit censura em zero

# E[Y_obs] via formula analitica de Tobit
xb = X_vec * beta_t
index = xb / sig_t
E_Y_tobit_formula = norm.cdf(index) * xb + sig_t * norm.pdf(index)

# Verificar com media amostral condicionada
print(f''\nE[Y_obs] Tobit (formula):  {np.mean(E_Y_tobit_formula):.4f}'')
print(f''E[Y_obs] Tobit (amostral): {np.mean(Y_obs):.4f}'')
prop_censurada = np.mean(Y_obs == 0)
print(f''Proporcao censurada em 0:  {prop_censurada:.4f}'')

# Decompondo E[Y] = E[Y | Y>0] * P(Y>0) + 0 * P(Y=0) -- Lei da Probabilidade Total
E_cond    = np.mean(Y_obs[Y_obs > 0])
P_pos     = np.mean(Y_obs > 0)
print(f''\nDecomposicao (Lei Prob Total):'')
print(f''  E[Y|Y>0] = {E_cond:.4f}, P(Y>0) = {P_pos:.4f}'')
print(f''  E[Y] = E[Y|Y>0]*P(Y>0) + 0*P(Y=0) = {E_cond * P_pos:.4f}'')
