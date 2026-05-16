import numpy as np
import scipy.stats as stats

class MotorMQO:
    def __init__(self, y, X):
        self.y = y
        self.X = X
        self.n, self.k = X.shape
        
    def estimar(self):
        # 1. Beta = (X'X)^-1 X'y
        XtX_inv = np.linalg.inv(self.X.T @ self.X)
        self.beta = XtX_inv @ self.X.T @ self.y
        
        # 2. Residuos e Sigma2
        self.residuos = self.y - self.X @ self.beta
        self.sigma2 = (self.residuos.T @ self.residuos) / (self.n - self.k)
        
        # 3. Variancia de Beta
        self.var_beta = self.sigma2 * XtX_inv
        self.se_beta = np.sqrt(np.diag(self.var_beta))
        
        # 4. Estatisticas t e p-valores
        self.t_stats = self.beta / self.se_beta
        self.p_values = [2 * (1 - stats.t.cdf(np.abs(t), df=self.n-self.k)) for t in self.t_stats]
        
        return self.beta, self.se_beta, self.p_values

# Exemplo de Uso
# X = np.column_stack([np.ones(100), np.random.randn(100)])
# y = 2 + 1.5 * X[:, 1] + np.random.normal(0, 1, 100)
# modelo = MotorMQO(y, X)
# beta, se, p = modelo.estimar()
