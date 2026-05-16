import statsmodels.api as sm

def estimar_probit_completo(y, X):
    X = sm.add_constant(X)
    modelo = sm.Probit(y, X).fit()
    print(modelo.summary())
    
    # Calculo dos Efeitos Marginais Medios (AME)
    margeff = modelo.get_margeff(at='overall')
    print(''\nEfeitos Marginais (AME):'')
    print(margeff.summary())
    
    return modelo
