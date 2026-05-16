import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import statsmodels.api as sm

np.random.seed(42)
n = 1000
k = 20
X = np.random.normal(0, 1, (n, k))
# D eh funcao nao-linear de X
D = np.sin(X[:, 0]) + 0.5 * X[:, 1]**2 + np.random.normal(0, 0.5, n)
# Y eh funcao de D e nao-linear de X
theta_true = 2.5
Y = theta_true * D + np.exp(X[:, 2]) + X[:, 3] * X[:, 4] + np.random.normal(0, 1, n)

# Algoritmo DML com 2-Fold Cross-Fitting
kf = KFold(n_splits=2, shuffle=True)
res_Y = np.zeros(n)
res_D = np.zeros(n)

for train_idx, test_idx in kf.split(X):
    # Modelos de Nuisance (Random Forest)
    model_y = RandomForestRegressor(n_estimators=100)
    model_d = RandomForestRegressor(n_estimators=100)
    
    model_y.fit(X[train_idx], Y[train_idx])
    model_d.fit(X[train_idx], D[train_idx])
    
    res_Y[test_idx] = Y[test_idx] - model_y.predict(X[test_idx])
    res_D[test_idx] = D[test_idx] - model_d.predict(X[test_idx])

# Regressao final dos residuos (Neyman Orthogonal)
dml_model = sm.OLS(res_Y, res_D).fit()
print(f"Efeito Causal Estimado (DML): {dml_model.params[0]:.4f}")
print(dml_model.conf_int())
