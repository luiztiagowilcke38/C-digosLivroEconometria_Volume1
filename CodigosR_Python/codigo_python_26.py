import numpy as np
from scipy.linalg import inv
from scipy.optimize import minimize

np.random.seed(1)
n = 1000
z1, z2 = np.random.normal(0, 1, n), np.random.normal(0, 1, n)
u = np.random.normal(0, 1, n)
x = 0.5*z1 + 0.5*z2 + 0.5*u + np.random.normal(0, 0.5, n)
y = 2.0*x + u  # beta verdadeiro = 2.0

Z = np.column_stack([np.ones(n), z1, z2])
X = np.column_stack([np.ones(n), x])

# Passo 1: GMM com W = (Z'Z/n)^-1 (Equivale a 2SLS se sobreidentificado)
W1 = inv(Z.T @ Z / n)
def gmm_obj(beta, Y, X, Z, W):
    moments = Z.T @ (Y - X @ beta) / len(Y)
    return moments.T @ W @ moments

res1 = minimize(gmm_obj, [0, 0], args=(y, X, Z, W1))
beta_step1 = res1.x

# Passo 2: Calcular matriz de ponderacao otima S^-1
e = y - X @ beta_step1
g = Z * e[:, None]
S = (g.T @ g) / n
W2 = inv(S)

res2 = minimize(gmm_obj, beta_step1, args=(y, X, Z, W2))
beta_gmm = res2.x
print(f"Beta GMM (2-step): {beta_gmm[1]:.4f}")

# Teste de Sobreidentificacao (Hansen J-test)
J_stat = n * gmm_obj(beta_gmm, y, X, Z, W2)
from scipy.stats import chi2
p_val_j = 1 - chi2.cdf(J_stat, df=Z.shape[1] - X.shape[1])
print(f"J-stat: {J_stat:.4f}, p-valor: {p_val_j:.4f} (H0: instrumentos validos)")
