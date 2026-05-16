import numpy as np
import pandas as pd
from scipy.stats import norm

np.random.seed(99)
n = 3000

# DAG: C -> X, C -> Y, X -> Y (C e' confounder)
# Efeito causal verdadeiro de X em Y = 2.0
C = np.random.normal(0, 1, n)     # Confounder observavel
X = 0.6 * C + np.random.normal(0, np.sqrt(1-0.36), n)
Y = 2.0 * X + 1.5 * C + np.random.normal(0, 1, n)

# (a) Estimativa MQO ingenua (viesada - ignora C)
beta_crude = np.cov(X, Y)[0,1] / np.var(X)
print(f''Estimativa ingénua (sem controle C): {beta_crude:.3f} (verdadeiro: 2.0)'')

# (b) Criterio Back-Door: Controlar por C elimina a porta dos fundos C->X<-...
# Ajustar por C e exato gracias a linearidade da equacao
from numpy.linalg import lstsq
X_mat = np.column_stack([np.ones(n), X, C])
beta_adj, *_ = lstsq(X_mat, Y, rcond=None)
print(f''Estimativa ajustada (controle C):    {beta_adj[1]:.3f} (verdadeiro: 2.0)'')

# (c) Verificacao formal via formula G-computation (integracao sobre dist. C)
# E[Y | do(X=x)] = integral E[Y|X=x, C=c] P(C=c) dc
# Para X continuo, o ATE de X+1 vs X e' simplesmente beta_adj[1]
n_boot = 2000
betas_boot = np.empty(n_boot)
for b in range(n_boot):
    idx = np.random.choice(n, n, replace=True)
    X_b = np.column_stack([np.ones(n), X[idx], C[idx]])
    betas_boot[b] = lstsq(X_b, Y[idx], rcond=None)[0][1]
ci_low, ci_high = np.percentile(betas_boot, [2.5, 97.5])
print(f''Bootstrap 95% CI: [{ci_low:.3f}, {ci_high:.3f}]'')
