import numpy as np

def simular_heston(S0, v0, mu, kappa, v_bar, sigma, rho, T, n_steps):
    dt = T / n_steps
    S = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)
    S[0], v[0] = S0, v0
    
    # Gerar Brownianos Correlacionados
    Z1 = np.random.normal(0, 1, n_steps)
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, n_steps)
    
    for t in range(n_steps):
        # Euler-Maruyama para Variancia (truncated at 0)
        v[t+1] = v[t] + kappa*(v_bar - max(v[t], 0))*dt + \
                 sigma*np.sqrt(max(v[t], 0))*np.sqrt(dt)*Z2[t]
        
        # Euler-Maruyama para Preco
        S[t+1] = S[t] + mu*S[t]*dt + np.sqrt(max(v[t], 0))*S[t]*np.sqrt(dt)*Z1[t]
        
    return S, v
