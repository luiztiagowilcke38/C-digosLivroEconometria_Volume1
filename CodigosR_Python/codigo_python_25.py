import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from scipy.linalg import solve

np.random.seed(42)

# =========================================================
# PARTE A: MQO como KKT - Regressao Ridge com restricao ||beta|| <= t
# =========================================================
# min (y - X*beta)' (y - X*beta) + lambda * beta'*beta
# Equivale a min ||beta||^2 s.a. ||y - Xb||^2 <= sigma0^2
# As condicoes KKT mostram que beta_ridge = (X'X + lambda*I)^-1 X'y

n, k = 200, 4
X   = np.column_stack([np.ones(n), np.random.randn(n, k-1)])
beta_t = np.array([1.0, 2.0, -1.5, 0.8])
y   = X @ beta_t + np.random.normal(0, 1.5, n)

# Ridge via formula fechada (solucao KKT analitica)
def ridge_kkt(X, y, lam):
    n, k = X.shape
    # Gradiente da Lagrangeana = 0: 2X'Xb - 2X'y + 2*lam*b = 0
    # => (X'X + lam*I) b = X'y
    A = X.T @ X + lam * np.eye(k)
    return solve(A, X.T @ y)

print(''--- Ridge via KKT(formula fechada) ---'')
for lam in [0, 0.1, 1.0, 10.0]:
    b = ridge_kkt(X, y, lam)
    resid = y - X @ b
    print(f''  lambda={lam:5.1f}: beta={np.round(b,3)}  ||e||^2={resid@resid:.2f}'')

# =========================================================
# PARTE B: LASSO via KKT - Condicao de Subgradiente
# =========================================================
# Condicao KKT do LASSO: X'(y - Xb) = lam * sign(b)  [para b_j != 0]
#                        |X_j'(y - Xb)| <= lam         [para b_j == 0]
# Verificamos essa condicao apos estimar com scipy

from sklearn.linear_model import Lasso

lam_lasso = 0.5
lasso = Lasso(alpha=lam_lasso, fit_intercept=False, max_iter=10000)
lasso.fit(X, y)
b_lasso = lasso.coef_

resid_l = y - X @ b_lasso
grad_ll = X.T @ resid_l  # = X'(y - Xb)

print(''\n--- Verificacao KKT do LASSO ---'')
print(f''lambda = {lam_lasso}'')
for j in range(k):
    if np.abs(b_lasso[j]) > 1e-6:
        kkt_err = np.abs(grad_ll[j]) - lam_lasso * np.abs(np.sign(b_lasso[j]))
        print(f''  beta[{j}]={b_lasso[j]:7.4f} != 0: |grad_j - lam*sign|={kkt_err:.2e} (deve ~0)'')
    else:
        kkt_ok = np.abs(grad_ll[j]) <= lam_lasso + 1e-6
        print(f''  beta[{j}]={b_lasso[j]:7.4f} =  0: |grad_j|={np.abs(grad_ll[j]):.4f} <= {lam_lasso} ? {kkt_ok}'')

# =========================================================
# PARTE C: MQO com restricao de igualdade - Multiplicador de Lagrange
# =========================================================
# min (y-Xb)'(y-Xb)  s.a. R*b = r   (ex: beta_2 + beta_3 = 0)
# Solucao KKT: b* = b_ols - (X'X)^-1 R' [R(X'X)^-1 R']^-1 (R*b_ols - r)
# Multiplicador lambda* = [R(X'X)^-1 R']^-1 (R*b_ols - r)

R = np.array([[0, 0, 1, 1]])   # Restricao: beta_2 + beta_3 = 0
r = np.array([0.0])

XtX_inv = np.linalg.inv(X.T @ X)
b_ols   = XtX_inv @ X.T @ y
RXtXR   = R @ XtX_inv @ R.T

# Multiplicador de Lagrange
lambda_star = np.linalg.solve(RXtXR, R @ b_ols - r)
b_restrito  = b_ols - XtX_inv @ R.T @ lambda_star

print(''\n--- MQO com restricao R*beta = r (KKT Igualdade) ---'')
print(f''beta irrestrito:     {np.round(b_ols, 4)}'')
print(f''beta restrito:       {np.round(b_restrito, 4)}'')
print(f''Restricao R*b = {r[0]:.0f}: {R @ b_restrito}'')
print(f''Multiplicador KKT lambda* = {lambda_star[0]:.4f}'')
print(f''Interpretacao: custo marginal da restricao (SSE extra = {lambda_star[0]:.4f} por u.a.)'')

# =========================================================
# PARTE D: Verificando Folga Complementar (Complementary Slackness)
# =========================================================
# No Ridge, a restricao inativa (||b||^2 < t) implica mu*=0 (folga complementar)
# A restricao ativa implica mu*>0 e a restricao vale com igualdade

print(''\n--- Folga Complementar no Ridge (t fixo) ---'')
t_bound = 2.0  # restricao ||beta||^2 <= t
for lam in [0.01, 0.5, 5.0]:
    b_r     = ridge_kkt(X, y, lam)
    norm_b  = b_r @ b_r
    ativo   = norm_b >= t_bound - 0.01
    mu_kkt  = lam     # O multiplicador KKT equivale exatamente ao parametro lambda
    print(f''  lambda={lam:.2f}: ||b||^2={norm_b:.3f}  ativo={ativo}  mu*={mu_kkt:.2f}  ''
          f''mu**(||b||^2-t)={mu_kkt*(norm_b-t_bound):.4f} (folga compl.)'')
