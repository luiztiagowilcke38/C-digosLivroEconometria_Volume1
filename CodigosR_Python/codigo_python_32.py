from sklearn.linear_model import LassoCV
import statsmodels.api as sm
import numpy as np

# Simulacao: 100 instrumentos, apenas 5 sao fortes
n, p = 500, 100
Z = np.random.normal(0, 1, (n, p))
D = Z[:, :5] @ np.array([1, 0.8, 0.6, 0.4, 0.2]) + np.random.normal(0, 1, n)
Y = 2.0 * D + np.random.normal(0, 1, n)

# 1. Primeiro Estagio via LassoCV (Selecao Automatica)
lasso_1st = LassoCV(cv=5).fit(Z, D)
D_hat = lasso_1st.predict(Z)

# 2. Segundo Estagio (MQ2E)
iv_model = sm.OLS(Y, sm.add_constant(D_hat)).fit()
print(f"Coeficiente IV (Lasso-IV): {iv_model.params[1]:.4f}")
