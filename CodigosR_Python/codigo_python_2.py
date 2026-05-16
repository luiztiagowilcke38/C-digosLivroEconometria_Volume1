import numpy as np
from scipy.linalg import inv

def arellano_bond_gmm(y, x, n, t):
    """
    Simples GMM-Difference (1st step) para y_it = rho*y_{i,t-1} + beta*x_it + alpha_i + e_it
    n: numero de individuos, t: periodos totais
    """
    # 1. Transformacao em Primeira Diferenca (t-2 observacoes uteis por individuo)
    dy = []
    dx = []
    for i in range(n):
        idx = i*t
        y_i = y[idx : idx+t]
        x_i = x[idx : idx+t]
        dy.append(np.diff(y_i)[1:]) # y_t - y_{t-1} para t >= 3
        dx_i = np.column_stack([np.diff(y_i)[:-1], np.diff(x_i)[1:]])
        dx.append(dx_i)
    
    DY = np.concatenate(dy) # (n*(t-2)) x 1
    DX = np.concatenate(dx) # (n*(t-2)) x 2
    
    # 2. Matriz de Instrumentos Z (Niveis defasados)
    Z_list = []
    for i in range(n):
        y_i = y[i*t : i*t+t]
        # Para t=3, instrumento eh y_i1. Para t=4, instrumentos [y_i1, y_i2] etc.
        # Simplificacao: apenas y_{i, t-2}
        z_i = y_i[:-2] 
        Z_list.append(np.diag(z_i))
    
    Z = np.block(Z_list)
    
    # 3. Estimador GMM: beta = (X'Z (Z'Z)^-1 Z'X)^-1 X'Z (Z'Z)^-1 Z'Y
    Pz = Z @ inv(Z.T @ Z) @ Z.T
    beta_gmm = inv(DX.T @ Pz @ DX) @ DX.T @ Pz @ DY
    return beta_gmm

# Exemplo de simulacao:
n, t = 100, 10
rho = 0.5
beta = 1.2
y = np.zeros(n*t)
x = np.random.normal(0, 1, n*t)
for i in range(n):
    alpha_i = np.random.normal(0, 1)
    for s in range(1, t):
        idx = i*t + s
        y[idx] = rho*y[idx-1] + beta*x[idx] + alpha_i + np.random.normal(0, 0.5)

res = arellano_bond_gmm(y, x, n, t)
print(f"Estimativa GMM: rho={res[0]:.3f}, beta={res[1]:.3f}")
