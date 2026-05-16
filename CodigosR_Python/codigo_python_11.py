import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def pipeline_econometrico(path_data):
    # 1. Carregamento e Limpeza
    df = pd.read_csv(path_data)
    df = df.dropna(subset=['salario', 'educacao'])
    
    # 2. Criacao de variaveis (Ex: logaritmos)
    df['ln_salario'] = np.log(df['salario'])
    
    # 3. Estimacao de Modelos em Cascata
    m1 = smf.ols('ln_salario ~ educacao', data=df).fit()
    m2 = smf.ols('ln_salario ~ educacao + exper + idade', data=df).fit()
    
    # 4. Exportacao de Tabelas (Padrao Profissional)
    from statsmodels.iolib.summary2 import summary_col
    print(summary_col([m1, m2], stars=True))
