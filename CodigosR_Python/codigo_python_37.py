import numpy as np
import scipy.stats as stats
from scipy.linalg import cholesky

# 1. Definir Correlação de Cauda e Graus de Liberdade (Não Linear Multivariado)
corr_matrix = np.array([[1.0, 0.7], [0.7, 1.0]])
N_sim = 10000

# 2. Gerar Amostras Normais Correlacionadas via Cholesky Decomp
L = cholesky(corr_matrix, lower=True)
Z = np.random.normal(0, 1, size=(2, N_sim))
Y_norm = L @ Z

# 3. Transformada da Probabilidade Integral (TPI) - Transforma pra Uniforme [0,1]
U = stats.norm.cdf(Y_norm)

# 4. Inversão Marginal Customizada: Unificando uma Expo e uma T-Student(df=4)
# Margem A: Exponencial (riscos assimetricos tempo p/ falha)
asset_A = stats.expon.ppf(U[0, :], scale=1/0.5)

# Margem B: Fat Tail T-Student (retornos)
asset_B = stats.t.ppf(U[1, :], df=4, loc=0, scale=0.02)

print(f''Corr Spearman Oculto: {stats.spearmanr(asset_A, asset_B)[0]:.3f}'')
