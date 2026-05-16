import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

# Simulacao Domiciliar de Dados
np.random.seed(123)
N = 250
x_true = np.random.randn(N)
alpha_true, beta_true, sigma_true = 2.5, 1.2, 0.8
y_obs = alpha_true + beta_true * x_true + np.random.randn(N) * sigma_true

# Definicao Generativa Bayesiana Formidavel no PyMC (NUTS Sampling)
with pm.Model() as bayesian_model:
    # 1. Priori Formativas Fracas
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10)
    
    # Half-Cauchy estritamente positiva para Dispersao de variancia
    sigma = pm.HalfCauchy('sigma', beta=5)
    
    # 2. Likelihood Determinada
    mu = alpha + beta * x_true
    Y_pred = pm.Normal('Y_pred', mu=mu, sigma=sigma, observed=y_obs)
    
    # 3. Inicializacao Hamiltoniana de No-U-Turn Sampler (MCMC)
    trace = pm.sample(draws=2000, tune=1000, 
                      chains=4, return_inferencedata=True)

# Diagnosticando Arviz HDI (Highest Density Interval a 94%)
print(az.summary(trace, round_to=3))

# Plotando Convergencia (Trace Plots em Lag)
# az.plot_trace(trace)
# plt.show()
