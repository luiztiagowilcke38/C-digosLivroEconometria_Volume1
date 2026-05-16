import numpy as np
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor

def double_machine_learning(y, D, X):
    # D: tratamento, X: controles (alta dimensao)
    
    # 1. Aprender E[Y|X] e extrair residuos
    model_y = RandomForestRegressor(n_estimators=100)
    model_y.fit(X, y)
    res_y = y - model_y.predict(X)
    
    # 2. Aprender E[D|X] e extrair residuos
    model_d = RandomForestRegressor(n_estimators=100)
    model_d.fit(X, D)
    res_d = D - model_d.predict(X)
    
    # 3. Estimar efeito causal final
    res_d = res_d.reshape(-1, 1)
    final_model = LinearRegression().fit(res_d, res_y)
    
    return final_model.coef_[0]
