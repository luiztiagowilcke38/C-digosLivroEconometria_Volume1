import numpy as np
import pandas as pd
import statsmodels.api as sm

np.random.seed(42)
n = 5000

# 1. Gerar Confounding Oculto (U)
U = np.random.normal(10, 2, n)

# 2. X depende de U
X = 0.8 * U + np.random.normal(0, 1, n)

# 3. Mediador M depende APENAS de X (Condicao 2 do Front-Door)
M = 1.2 * X + np.random.normal(0, 1, n)

# 4. Y depende de M e do Confounding Oculto U (Condicao 3)
# Efeito Real de X -> Y via M: 1.2 * 2.5 = 3.0
Y = 2.5 * M + 1.5 * U + np.random.normal(0, 1, n)

df = pd.DataFrame({'X': X, 'M': M, 'Y': Y})

# --- ABORDAGEM 1: MQO INGENUO (Viesado por U) ---
model_naive = sm.OLS(df['Y'], sm.add_constant(df['X'])).fit()
print(f"Efeito Naive (X->Y): {model_naive.params['X']:.4f} (Viesado!)")

# --- ABORDAGEM 2: AJUSTE FRONT-DOOR ---
# Passo A: P(M|X) -> Efeito de X em M (isento de U)
stage1 = sm.OLS(df['M'], sm.add_constant(df['X'])).fit()
beta_xm = stage1.params['X']

# Passo B: P(Y|M, X) -> Efeito de M em Y controlando por X 
# (X bloqueia o caminho back-door M<-X<-U->Y)
stage2 = sm.OLS(df['Y'], sm.add_constant(df[['M', 'X']])).fit()
beta_my = stage2.params['M']

# Efeito Causal Estimado = beta_xm * beta_my
front_door_effect = beta_xm * beta_my
print(f"Efeito Front-Door:    {front_door_effect:.4f} (Real ~ 3.0)")
