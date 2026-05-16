import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank

# Simulacao de dados cointegrados
n = 250
x = np.cumsum(np.random.normal(0, 1, n))
y = 0.5 * x + np.random.normal(0, 1, n)
df = pd.DataFrame({'y': y, 'x': x})

# Escolha do Rank de Cointegracao
rank = select_coint_rank(df, det_order=0, k_ar_diff=1)
print(f"Rank sugerido: {rank.rank}")

# Estimacao do VECM
vecm_model = VECM(df, k_ar_diff=1, coint_rank=1, deterministic='co')
vecm_res = vecm_model.fit()
print(vecm_res.summary())
