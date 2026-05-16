import numpy as np
import scipy.stats as stats
from arch import arch_model

# Dados simulados de retornos (ou carregar dados reais)
np.random.seed(42)
n = 1000
ret = np.random.normal(0, 1, n) * 0.01

# Estimar GARCH(1,1) com distribuicao t-Student
am = arch_model(ret * 100, vol='Garch', p=1, q=1, dist='t')
res = am.fit(disp='off')

# Extrair variancia condicional e graus de liberdade
conditional_vol = res.conditional_volatility / 100
df_hat = res.params['nu']
mu_hat = res.params['mu'] / 100

# VaR e ES condicionais a 99%
alpha = 0.99
z_alpha = stats.t.ppf(alpha, df=df_hat)
z_es = stats.t.pdf(z_alpha, df=df_hat) / (1 - alpha)

VaR99 = -mu_hat - conditional_vol * z_alpha
ES99 = -mu_hat + conditional_vol * (z_es * df_hat / (df_hat - 1))

print(f'VaR 99 pct (ultimo dia): {VaR99[-1]*100:.4f} pct')
print(f'ES 99 pct (ultimo dia): {ES99[-1]*100:.4f} pct')

# Backtesting simples: Contagem de violacoes
violacoes = np.sum(-ret > VaR99)
print(f'Violacoes observadas: {violacoes} (esperado aprox. {int(n*(1-alpha))})')
