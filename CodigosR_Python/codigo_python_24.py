import numpy as np
from numpy.linalg import solve

np.random.seed(42)
n = 300

# Dados base: dois regressores relevantes
x1 = np.random.randn(n)
x2 = np.random.randn(n)
y  = 2.0 + 1.5*x1 - 0.8*x2 + np.random.normal(0, 1.5, n)

def ols_stats(X, y):
    '''Retorna: beta, y_hat, resid, R2, R2_adj, SST, SSR, SSE'''
    n, k = X.shape[0], X.shape[1] - 1  # -1 exclui intercepto
    beta  = solve(X.T @ X, X.T @ y)
    y_hat = X @ beta
    resid = y - y_hat
    SSE   = resid @ resid
    SST   = ((y - y.mean()) @ (y - y.mean()))
    SSR   = SST - SSE
    R2    = 1 - SSE/SST
    R2adj = 1 - (SSE/(n-k-1)) / (SST/(n-1))
    return beta, y_hat, resid, R2, R2adj, SST, SSR, SSE

# =========================================================
# PARTE A: R2 com 0, 1 e 2 regressores relevantes
# =========================================================
for label, cols in [(''nenhum regressor'', [np.ones(n)]),
                    (''so x1'',            [np.ones(n), x1]),
                    (''x1 e x2'',          [np.ones(n), x1, x2])]:
    X_m = np.column_stack(cols)
    _, _, _, R2, R2adj, SST, SSR, SSE = ols_stats(X_m, y)
    k_m = X_m.shape[1] - 1
    print(f''Modelo ({label}):  R2={R2:.4f}  R2adj={R2adj:.4f}  ''
          f''SST={SST:.2f}  SSR={SSR:.2f}  SSE={SSE:.2f}'')

print()

# =========================================================
# PARTE B: R2 NUNCA diminui - adicionando variaveis aleatorias
# =========================================================
print(''R2 ao adicionar variaveis aleatorias (deveriam ser irrelevantes):'')
X_base = np.column_stack([np.ones(n), x1, x2])
for extra in range(0, 10):
    if extra == 0:
        X_m = X_base
    else:
        X_m = np.column_stack([X_base] +
                               [np.random.randn(n) for _ in range(extra)])
    _, _, _, R2, R2adj, *_ = ols_stats(X_m, y)
    print(f''  k={X_m.shape[1]-1:2d} regressores: R2={R2:.5f}  ''
          f''R2adj={R2adj:.5f}  {'*** R2adj CAIU' if R2adj < 0 else ''}'')

# =========================================================
# PARTE C: Regressao Espuria - R2 alto sem relacao verdadeira
# =========================================================
print(''\n--- Regressao Espuria (Passeios Aleatorios Independentes) ---'')
T = 500
walk1 = np.cumsum(np.random.randn(T))   # Passeio 1
walk2 = np.cumsum(np.random.randn(T))   # Passeio 2 (independente!)

X_sp = np.column_stack([np.ones(T), walk1])
_, _, resid_sp, R2_sp, R2adj_sp, *_ = ols_stats(X_sp, walk2)
# Teste: se ha autocorrelacao nos residuos => sinal de regressao espuria
from scipy.stats import pearsonr
dw = np.sum(np.diff(resid_sp)**2) / np.sum(resid_sp**2)
print(f''  R2 da regressao espuria:    {R2_sp:.4f}  (alto, mas falso!)'')
print(f''  Durbin-Watson:              {dw:.4f}  (longe de 2 => autocorrelacao resid.)'')
print(f''  Conclusao: R2 alto + DW baixo = sinal classico de regressao espuria'')

# =========================================================
# PARTE D: R2 sem intercepto (problema de definicao)
# =========================================================
print(''\n--- R2 sem intercepto (armadilha) ---'')
X_no_int = x1.reshape(-1, 1)
beta_ni  = solve(X_no_int.T @ X_no_int, X_no_int.T @ y)
y_hat_ni = X_no_int @ beta_ni
resid_ni = y - y_hat_ni

SSE_ni    = resid_ni @ resid_ni
SST_usual = ((y - y.mean()) @ (y - y.mean()))   # usa media
SST_zero  = y @ y                                # sem intercepto: referencia = 0

R2_errado = 1 - SSE_ni / SST_usual  # Pode ser negativo!
R2_correto= 1 - SSE_ni / SST_zero   # Base correta para modelo sem constante
print(f''  R2 (formula com media, incorreto s/ intercepto): {R2_errado:.4f}'')
print(f''  R2 (formula base=0,   correto s/ intercepto):    {R2_correto:.4f}'')

# =========================================================
# PARTE E: Comparacao R2 entre log(y) e y (invalida!)
# =========================================================
print(''\n--- Incompatibilidade de R2 entre transformacoes ---'')
y_pos = np.exp(0.5 + 0.3*x1 + np.random.normal(0, 0.3, n))  # Y log-normal
log_y = np.log(y_pos)

X_m  = np.column_stack([np.ones(n), x1])
_, _, _, R2_log, *_ = ols_stats(X_m, log_y)
_, _, _, R2_lev, *_ = ols_stats(X_m, y_pos)
print(f''  R2 modelo log(Y) ~ X:  {R2_log:.4f}'')
print(f''  R2 modelo Y ~ X:       {R2_lev:.4f}'')
print(''  -> R2 nao e comparavel entre escalas diferentes de y!'')
