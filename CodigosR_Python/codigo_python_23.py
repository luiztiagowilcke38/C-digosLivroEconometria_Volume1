import numpy as np
from scipy.stats import multivariate_normal, norm, chi2
import scipy.linalg as la

np.random.seed(42)

# =========================================================
# PARTE A: Geracao e Densidade da Normal Multivariada
# =========================================================
k = 4
mu = np.array([1.0, -0.5, 2.0, 0.3])

# Gerando uma matriz de covariancia valida (positiva definida)
A = np.random.randn(k, k)
Sigma = A @ A.T / k + np.eye(k)  # Garante pos. definida
print(''Matriz Sigma:\n'', np.round(Sigma, 3))
print(''Autovalores (todos positivos):'', np.round(np.linalg.eigvalsh(Sigma), 3))

# Densidade em um ponto
x0 = np.zeros(k)
mvn = multivariate_normal(mean=mu, cov=Sigma)
print(f''\nDensidade em x=0: {mvn.pdf(x0):.6f}'')

# Calculo manual da densidade (formula acima)
diff     = x0 - mu
inv_Sig  = la.inv(Sigma)
det_Sig  = la.det(Sigma)
mahal2   = diff @ inv_Sig @ diff
f_manual = (1 / ((2*np.pi)**(k/2) * det_Sig**0.5)) * np.exp(-0.5 * mahal2)
print(f''Densidade manual:   {f_manual:.6f}'')
print(f''Distancia Mahalanobis: {np.sqrt(mahal2):.4f}'')

# =========================================================
# PARTE B: Decomposicao de Cholesky para Simulacao
# =========================================================
# Se L = chol(Sigma), entao X = mu + L @ Z com Z ~ N(0, I) => X ~ N(mu, Sigma)
L    = la.cholesky(Sigma, lower=True)
N    = 50_000
Z    = np.random.randn(k, N)
X_s  = mu[:, None] + L @ Z      # Simulados via Cholesky
print(''\nMedia amostral vs mu:'')
print(''  amostral:'', np.round(X_s.mean(axis=1), 3))
print(''  teorico: '', np.round(mu, 3))
print(''Maxima diferenca na Sigma estimada vs teorica:'',
      np.round(np.max(np.abs(np.cov(X_s) - Sigma)), 4))

# =========================================================
# PARTE C: Teorema de Cramer-Wold - Verificacao Numerica
# =========================================================
print(''\n--- Verificacao do Teorema de Cramer-Wold ---'')
# Para qualquer c, c'X deve ser N(c'mu, c'Sigma c)
for i in range(5):
    c = np.random.randn(k)
    c = c / np.linalg.norm(c)    # Normalizar para comparacao justa

    # Combinacao linear das amostras simuladas
    proj  = c @ X_s              # Escalar para cada amostra (N valores)
    mu_proj    = c @ mu
    sigma_proj = np.sqrt(c @ Sigma @ c)

    # Teste de Shapiro-Wilk sobre 5000 amostras da combinacao linear
    from scipy.stats import shapiro
    stat, p_val = shapiro(proj[:5000])
    print(f''  c{i+1}: E[c'X]={mu_proj:.3f} (amostral={proj.mean():.3f}) | ''
          f''Std={sigma_proj:.3f} (amostral={proj.std():.3f}) | ''
          f''Shapiro p={p_val:.3f} {'OK-Normal' if p_val > 0.05 else 'Atencao!'}'')

# =========================================================
# PARTE D: Distribuicoes Condicionais e Marginais
# =========================================================
# Particionando: X = [X1 (dim 2), X2 (dim 2)]
# X1|X2=x2 ~ N(mu1 + Sigma12 Sigma22^-1 (x2-mu2), Sigma11 - Sigma12 Sigma22^-1 Sigma21)
mu1, mu2         = mu[:2], mu[2:]
S11, S12         = Sigma[:2, :2], Sigma[:2, 2:]
S21, S22         = Sigma[2:, :2], Sigma[2:, 2:]
inv_S22          = la.inv(S22)

x2_obs           = np.array([1.5, 0.0])   # Valor observado de X2
mu_cond          = mu1 + S12 @ inv_S22 @ (x2_obs - mu2)
Sigma_cond       = S11 - S12 @ inv_S22 @ S21

print(''\n--- Distribuicao Condicional X1 | X2=x2 ---'')
print(f''Media condicional:     {np.round(mu_cond, 4)}'')
print(f''Cov. condicional:\n{np.round(Sigma_cond, 4)}'')

# Verificar empiricamente: filtrar amostras com X2 proximo de x2_obs
tol = 0.15
mascara = np.all(np.abs(X_s[2:, :] - x2_obs[:, None]) < tol, axis=0)
X1_cond_amostral = X_s[:2, mascara]
print(f''Media condicional (amostral, n={mascara.sum()}):'', np.round(X1_cond_amostral.mean(axis=1), 4))

# =========================================================
# PARTE E: Forma Quadratica e Distribuicao Qui-Quadrado
# =========================================================
# Se X ~ N(mu, Sigma), entao (X-mu)' Sigma^{-1} (X-mu) ~ chi2(k)
X_s2     = X_s.T   # N x k
diffs    = X_s2 - mu
mahal_sq = np.einsum('ni,ij,nj->n', diffs, inv_Sig, diffs)

# Testar: Seja o empirico compativel com chi2(k=4)?
from scipy.stats import kstest
stat_ks, p_ks = kstest(mahal_sq, 'chi2', args=(k,))
print(f''\nDistancia Mahalanobis^2 ~ chi2({k}):'')
print(f''  Media amostral: {mahal_sq.mean():.3f} (teorico: {k})'')
print(f''  KS p-valor: {p_ks:.4f} ({'Aceita chi2' if p_ks > 0.05 else 'Rejeita'})'')
