import numpy as np

class BayesianDLM:
    def __init__(self, m0, C0, F, G, V, W):
        self.m = m0.copy()  # Media a priori do estado
        self.C = C0.copy()  # Variancia a priori do estado
        self.F = F          # Matriz de observacao
        self.G = G          # Matriz de transicao
        self.V = V          # Variancia do erro de observacao
        self.W = W          # Variancia do erro de estado
        self.history_m = [m0.copy()]
        self.history_C = [C0.copy()]

    def update(self, y_t):
        # 1. PREVISAO (Prior One-Step-Ahead)
        a_t = self.G @ self.m           # Media do estado previsto
        R_t = self.G @ self.C @ self.G.T + self.W  # Covariancia prevista

        # 2. VEROSSIMILHANÇA Preditiva Normal
        f_t = self.F.T @ a_t            # Predicao de y
        Q_t = self.F.T @ R_t @ self.F + self.V   # Variancia preditiva

        # 3. GANHO DE KALMAN (Atualização Ótima)
        A_t = R_t @ self.F / Q_t       # Ganho de Kalman

        # 4. ATUALIZACAO POSTERIOR
        e_t = y_t - f_t                # Erro de previsao
        self.m = a_t + A_t * e_t       # Media posterior
        self.C = R_t - A_t * Q_t * A_t.T  # Covariancia posterior
        self.history_m.append(self.m.copy())
        self.history_C.append(self.C.copy())

# Simulando passeio aleatorio localmente estacionario
np.random.seed(5)
T = 200
theta_true = np.cumsum(np.random.normal(0, 0.5, T))
y_obs = theta_true + np.random.normal(0, 1.0, T)

# Inicializar DLM scalar
F = np.array([[1.0]])
G = np.array([[1.0]])
V = np.array([[1.0]])
W = np.array([[0.25]])
m0 = np.array([[0.0]])
C0 = np.array([[10.0]])

dlm = BayesianDLM(m0, C0, F, G, V, W)
for yt in y_obs:
    dlm.update(np.array([[yt]]))

# Recuperar serie de medianas filtradas
filtered = [m[0,0] for m in dlm.history_m[1:]]
print(f''RMSE Filtro Kalman: {np.sqrt(np.mean((np.array(filtered)-theta_true)**2)):.4f}'')
