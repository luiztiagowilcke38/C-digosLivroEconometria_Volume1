import numpy as np
import scipy.linalg as la

# Dados sinteticos com 3 regressores (n=500)
np.random.seed(42)
n, k = 500, 3
X_raw = np.random.randn(n, k)
X = np.column_stack([np.ones(n), X_raw])  # Adiciona constante
beta_true = np.array([2.0, 1.5, -0.8, 0.5])
y = X @ beta_true + np.random.normal(0, 1.5, n)

# SOLUCAO 1: Formula Fechada Analitica (X'X)^{-1} X'y
XtX = X.T @ X
Xty = X.T @ y
beta_ols = la.solve(XtX, Xty)  # Mais estavel que np.linalg.inv(XtX) @ Xty
print(''MQO (formula fechada):'', beta_ols)

# SOLUCAO 2: Decomposicao QR (numericamente mais estavel)
Q, R = la.qr(X, mode='economic')
beta_qr = la.solve_triangular(R, Q.T @ y)
print(''MQO (QR):             '', beta_qr)

# SOLUCAO 3: SVD (controla multicolinearidade)
U, s, Vt = la.svd(X, full_matrices=False)
beta_svd = Vt.T @ np.diag(1/s) @ U.T @ y
print(''MQO (SVD):            '', beta_svd)

# VERIFICACAO: Projecao ortogonal - residuos perpendicualres a X
residuos = y - X @ beta_ols
print(f''\nVerificacao ortogonalidade (deve ser ~0): {X.T @ residuos}'')

# MATRIX HAT e Leverage
H = X @ la.inv(XtX) @ X.T
leverage = np.diag(H)

# Residuos Studentizados Internamente
sigma2 = (residuos @ residuos) / (n - k - 1)
se_residuals = np.sqrt(sigma2 * (1 - leverage))
stud_residuals = residuos / se_residuals
print(f''\nMax Leverage: {leverage.max():.4f}, outliers acima de {2*(k+1)/n:.4f}'')
print(f''Residuos outlier (|r|>3): {np.sum(np.abs(stud_residuals) > 3)} observacoes'')
