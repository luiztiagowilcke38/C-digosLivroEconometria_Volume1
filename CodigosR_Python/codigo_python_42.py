import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white, het_goldfeldquandt

def diagnostico_variancia(y, X):
    # Adicionar o intercepto se nao houver
    X = sm.add_constant(X)
    modelo = sm.OLS(y, X).fit()
    
    # 1. Teste de White
    white_test = het_white(modelo.resid, modelo.model.exog)
    labels = ['Lagrange Multiplier statistic', 'p-value', 'f-value', 'f p-value']
    print(''\nTeste de White:'')
    print(dict(zip(labels, white_test)))
    
    # 2. Teste de Goldfeld-Quandt
    gq_test = het_goldfeldquandt(y, X)
    print(''\nTeste Goldfeld-Quandt:'')
    print(dict(zip(['F-statistic', 'p-value'], gq_test)))
