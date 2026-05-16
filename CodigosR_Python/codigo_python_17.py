import numpy as np
from numpy.linalg import solve, inv

np.random.seed(123)
n = 1000
# Gerando um instrumento Z e variavel endogena X
U = np.random.normal(0, 1, n)
Z = np.random.normal(0, 1, n)

# X correlacionado com erro U (endogeneidade)
X_endo = 0.7 * Z + 0.5 * U + np.random.normal(0, 0.5, n)
Y = 2.0 * X_endo + 3.0 * U + np.random.normal(0, 1, n)

# Matrizes com constante
X_mat = np.column_stack([np.ones(n), X_endo])
Z_mat = np.column_stack([np.ones(n), Z])

# ESTAGIO 1: Projetar X_endo no espaco de Z
Pz = Z_mat @ inv(Z_mat.T @ Z_mat) @ Z_mat.T
X_hat = Pz @ X_mat  # X projetado

# ESTAGIO 2: Regredir Y em X_hat
beta_iv = solve(X_hat.T @ X_mat, X_hat.T @ Y)
print(f''Estimativa 2SLS: alpha={beta_iv[0]:.3f}, beta={beta_iv[1]:.3f} (verdadeiro=2.0)'')

# Comparacao com MQO viesado
beta_ols_biased = solve(X_mat.T @ X_mat, X_mat.T @ Y)
print(f''MQO viesado:     alpha={beta_ols_biased[0]:.3f}, beta={beta_ols_biased[1]:.3f}'')

# DIAGNOSTICO: Estatistica F do 1o estagio (Teste de Instrumento Fraco)
# Convencao: F < 10 => instrumento fraco (Stock & Yogo)
X_endo_col = X_endo.reshape(-1,1)
X_1st = np.column_stack([np.ones(n), Z])
beta_1st = solve(X_1st.T @ X_1st, X_1st.T @ X_endo)
resid_1st = X_endo - X_1st @ beta_1st
RSS_1st = resid_1st @ resid_1st
resid_rest = X_endo - X_endo.mean()
TSS_1st = resid_rest @ resid_rest
F_stat = ((TSS_1st - RSS_1st) / 1) / (RSS_1st / (n - 2))
print(f''\nEstatistica F 1o estagio: {F_stat:.2f} ({'Forte' if F_stat > 10 else 'FRACO!'})'')

# Matriz de variancia assintotica do 2SLS
e_iv = Y - X_mat @ beta_iv
sigma2_iv = (e_iv @ e_iv) / (n - X_mat.shape[1])
Var_iv = sigma2_iv * inv(X_hat.T @ X_mat)
se_iv = np.sqrt(np.diag(Var_iv))
t_iv = beta_iv / se_iv
print(f''\nErros-padrao 2SLS: {se_iv}'')
print(f''t-estatisticas:    {t_iv}'')
