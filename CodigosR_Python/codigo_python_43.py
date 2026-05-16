import numpy as np
import statsmodels.api as sm
from scipy.stats import norm

def heckman_two_step(y, X, d, W):
    # Passos 1: Probit de Selecao para obter gamma_hat
    selection_model = sm.Probit(d, W).fit()
    gamma_hat = selection_model.params
    
    # Passo 2: Calcular a Razao Inversa de Mills
    z_hat = W @ gamma_hat
    mills = norm.pdf(z_hat) / norm.cdf(z_hat)
    
    # Passo 3: MQO de y em X e Mills para os selecionados (d=1)
    X_augmented = np.column_stack([X[d==1], mills[d==1]])
    y_selected = y[d==1]
    
    outcome_model = sm.OLS(y_selected, X_augmented).fit()
    
    beta_hat = outcome_model.params[:-1]
    rho_sigma = outcome_model.params[-1] # Coeficiente da Mills
    
    return beta_hat, rho_sigma
