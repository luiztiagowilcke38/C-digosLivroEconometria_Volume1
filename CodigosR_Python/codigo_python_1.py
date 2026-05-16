import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

# Simulando Dados RDD - Uma descontinuidade C aos 50 anos de idade
c = 50
np.random.seed(10)
age = np.random.uniform(30, 70, 500)
# Regressao linear base, mas havera um ''jump'' de 15 pontos em y no limiar
y = 100 + 2*age + np.where(age >= c, 15, 0) + np.random.normal(0, 5, 500)

tratado = (age >= c).astype(int)

# Escolha Local de Bandwidth k=8
h = 8
subset = (age >= c - h) & (age <= c + h)

X = age[subset]
y_s = y[subset]
T = tratado[subset]

# Criar Interacao entre Tratamento e Distancia
dist = X - c
# Ponderacao Triangular Kernel : k(u) = 1 - abs(u)
weights = 1 - np.abs(dist / h)

# Matriz Design X = [1, T, dist, T*dist]
df_rd = pd.DataFrame({'Y': y_s, 'T': T, 'dist': dist, 'T_dist': T*dist})
X_rd = sm.add_constant(df_rd[['T', 'dist', 'T_dist']])

model_wls = sm.WLS(df_rd['Y'], X_rd, weights=weights)
results = model_wls.fit(cov_type='HC3')
print(results.summary())
print(f''O Efeito Causal Estimado (tau) eh de: {results.params['T']:.4f}'')
