import numpy as np
from scipy.optimize import linprog

np.random.seed(7)
n, p = 200, 2
X = np.column_stack([np.ones(n), np.random.randn(n)])
beta_true = np.array([1.0, 2.5])
y = X @ beta_true + 0.5 * np.random.normal(0, 1, n) * (1 + np.abs(np.random.randn(n)))
tau = 0.75  # Mediana superior (3o quartil)

# Regressao Quantilica = Programacao Linear
# Variaveis: [beta (p), u+ (n), u- (n)]
# min tau * sum(u+) + (1-tau) * sum(u-)
# s.t.: X beta + u+ - u- = y, u+ >= 0, u- >= 0
c = np.concatenate([np.zeros(p), tau*np.ones(n), (1-tau)*np.ones(n)])

A_eq = np.hstack([X, np.eye(n), -np.eye(n)])
b_eq = y

bounds = [(None, None)] * p + [(0, None)] * (2 * n)

result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
beta_qr_lp = result.x[:p]
print(f''Quantil tau={tau}: {beta_qr_lp}'')

# Comparacao com statsmodels
import statsmodels.formula.api as smf
import pandas as pd
df = pd.DataFrame({'y': y, 'x': X[:, 1]})
mod_smf = smf.quantreg('y ~ x', data=df).fit(q=tau)
print(f''statsmodels QR:   {mod_smf.params.values}'')
