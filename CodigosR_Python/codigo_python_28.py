import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, select_coint_rank

# Simulando duas series cointegradas
np.random.seed(42)
n = 500
common_trend = np.cumsum(np.random.normal(0, 1, n))
y1 = 0.8 * common_trend + np.random.normal(0, 0.5, n)
y2 = 1.2 * common_trend + np.random.normal(0, 0.5, n)
df = pd.DataFrame({'y1': y1, 'y2': y2})

# 1. Selecao do Lag (VAR em niveis)
lag_selection = select_order(df, maxlags=10, deterministic="n")
print(lag_selection.summary())

# 2. Teste de Cointegracao de Johansen
rank_test = select_coint_rank(df, det_order=0, k_ar_diff=1, method="trace")
print(f"Rank de cointegracao sugerido: {rank_test.rank}")

# 3. Estimacao do VECM
model = VECM(df, k_ar_diff=1, coint_rank=1, deterministic="n")
results = model.fit()
print(results.summary())

# Vetor de Cointegracao (Beta) e Ajuste (Alpha)
print(f"Beta: {results.beta.flatten()}")
print(f"Alpha: {results.alpha.flatten()}")
