import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects

# Simulacao de Painel
n, t = 100, 5
ids = np.repeat(np.arange(n), t)
time = np.tile(np.arange(t), n)
df = pd.DataFrame({'id': ids, 'time': time})
df['x'] = np.random.randn(n*t)
alpha = dict(zip(np.arange(n), np.random.normal(0, 1, n)))
df['alpha'] = df['id'].map(alpha)
# Endogeneidade: x correlacionado com alpha
df['x'] = df['x'] + 0.5*df['alpha']
df['y'] = 2.0*df['x'] + df['alpha'] + np.random.normal(0, 1, n*t)

df = df.set_index(['id', 'time'])

# 1. Fixed Effects
fe_mod = PanelOLS.from_formula('y ~ x + EntityEffects', data=df).fit()
print(f"FE beta: {fe_mod.params['x']:.4f}")

# 2. Mundlak: Incluir media de x
df['x_mean'] = df.groupby('id')['x'].transform('mean')
mundlak_mod = sm.OLS(df['y'], sm.add_constant(df[['x', 'x_mean']])).fit()
print(f"Mundlak beta: {mundlak_mod.params['x']:.4f}")
print(f"Mundlak gamma (p-valor): {mundlak_mod.pvalues['x_mean']:.4f}")
