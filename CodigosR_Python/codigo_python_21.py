import numpy as np
from numpy.linalg import solve, inv

np.random.seed(42)
n = 500

# Gerando regressores: X1 (controles) e X2 (variavel de interesse)
X1 = np.column_stack([np.ones(n), np.random.randn(n), np.random.randn(n)])  # k1=3
X2 = np.column_stack([np.random.randn(n), np.random.randn(n)])              # k2=2

# Gerando Y com coeficientes verdadeiros conhecidos
beta1_true = np.array([1.0, -0.5, 0.8])
beta2_true = np.array([2.0, -1.3])
y = X1 @ beta1_true + X2 @ beta2_true + np.random.normal(0, 1, n)

# === METODO 1: Regressao MQO Completa ===
X_full = np.hstack([X1, X2])
beta_full = solve(X_full.T @ X_full, X_full.T @ y)
beta2_full = beta_full[3:]   # Ultimos k2=2 coeficientes
print(''MQO Completo (beta2):'', beta2_full)

# === METODO 2: FWL - Usando o Aniquilador M1 ===

# Passo A: Construir M1 = I - X1 @ (X1'X1)^{-1} @ X1'
P1 = X1 @ solve(X1.T @ X1, X1.T)   # Projetor ortogonal em span(X1)
M1 = np.eye(n) - P1                  # Aniquilador de X1

# Verificar propriedades do aniquilador
print(f''\nVerificacoes de M1:'')
print(f''  Idempotente M1^2 = M1? {np.allclose(M1 @ M1, M1)}'')
print(f''  Simetrico M1=M1'?      {np.allclose(M1, M1.T)}'')
print(f''  Aniquila X1: max|M1 X1|= {np.max(np.abs(M1 @ X1)):.2e}'')

# Passo B: Parcializar y e X2 em relacao a X1
y_tilde  = M1 @ y    # Residuos de y em X1
X2_tilde = M1 @ X2   # Residuos de X2 em X1

# Passo C: Regredir y_tilde em X2_tilde
beta2_fwl = solve(X2_tilde.T @ X2_tilde, X2_tilde.T @ y_tilde)
print(''\nFWL (beta2):'', beta2_fwl)

# === VERIFICACAO: Os dois sao identicos? ===
print(f''\nDiferenca maxima |beta2_full - beta2_fwl|: {np.max(np.abs(beta2_full - beta2_fwl)):.2e}'')
print(''FWL CONFIRMADO!'' if np.allclose(beta2_full, beta2_fwl, atol=1e-8) else ''ERRO!'')

# === METODO 3: Via Regressoes Auxiliares (mais intuitivo didaticamente) ===
# Residuos de y em X1
from numpy.linalg import lstsq
y_res, *_ = lstsq(X1, y, rcond=None)
y_res = y - X1 @ y_res  # equivale a M1 @ y

# Residuos de cada coluna de X2 em X1
X2_res = np.column_stack([
    X2[:, j] - X1 @ lstsq(X1, X2[:, j], rcond=None)[0]
    for j in range(X2.shape[1])
])
beta2_aux = solve(X2_res.T @ X2_res, X2_res.T @ y_res)
print(''\nFWL via regresoes auxiliares (beta2):'', beta2_aux)

# === INSIGHT GEOMETRICO: cos(angulo) entre y_tilde e espaco de X2_tilde ===
R2_parcial = 1 - np.var(y_tilde - X2_tilde @ beta2_fwl) / np.var(y_tilde)
print(f''\nR2 parcial de X2 apos parcializar X1: {R2_parcial:.4f}'')
