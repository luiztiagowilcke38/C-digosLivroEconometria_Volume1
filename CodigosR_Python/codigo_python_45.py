import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simular_propriedades_bootstrap(n_obs=50, n_boot=2000):
    # Gerando dados: y = 2 + 1.5x + e
    x = np.random.normal(5, 2, n_obs)
    y = 2 + 1.5 * x + np.random.normal(0, 5, n_obs)
    
    # Estimador MQO Original
    beta_hat = np.polyfit(x, y, 1)[0]
    
    # Algoritmo de Bootstrap para Variancia de Beta
    betas_boot = []
    for _ in range(n_boot):
        indices = np.random.choice(range(n_obs), n_obs, replace=True)
        x_b, y_b = x[indices], y[indices]
        betas_boot.append(np.polyfit(x_b, y_b, 1)[0])
        
    se_boot = np.std(betas_boot)
    print(f''Beta original: {beta_hat:.4f}'')
    print(f''Erro Padrao Bootstrap: {se_boot:.4f}'')
    
    plt.hist(betas_boot, bins=50, color='lightgreen', edgecolor='black')
    plt.title(''Distribuicao Bootstrap de Beta'')
    plt.show()

def calcular_mse_simulado(estimador_func, theta_real, n_it=5000):
    # Teste de vies e variancia
    estimativas = [estimador_func() for _ in range(n_it)]
    vies = np.mean(estimativas) - theta_real
    variancia = np.var(estimativas)
    mse = variancia + vies**2
    return vies, variancia, mse
