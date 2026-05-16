import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def simular_metodo_delta(n_obs=1000, n_sims=5000):
    # g(theta) = theta^2. Se theta_hat -> N(mu, sigma^2/n)
    # entao g(theta_hat) -> N(mu^2, (2*mu)^2 * sigma^2/n)
    mu, sigma = 2, 1
    estimativas = []
    
    for _ in range(n_sims):
        x = np.random.normal(mu, sigma, n_obs)
        theta_hat = np.mean(x)
        estimativas.append(theta_hat**2)
        
    # Var observada vs Teorica do Metodo Delta
    var_delta = (2*mu)**2 * (sigma**2 / n_obs)
    print(f''Variancia Simulada: {np.var(estimativas)}'')
    print(f''Variancia Delta: {var_delta}'')
    
    plt.hist(estimativas, bins=50, density=True)
    plt.title(''Metodo Delta em Acao: g(theta) = theta^2'')
    plt.show()

def verificar_lgn_forte(n_max=10000):
    x = np.random.standard_t(df=1, size=n_max) # Cauchy tem media indefinida
    media_acum = np.cumsum(x) / np.arange(1, n_max + 1)
    
    plt.plot(media_acum)
    plt.axhline(0, color='r', ls='--')
    plt.title(''Falha da LGN Forte para Distribuicao Cauchy'')
    plt.show()
