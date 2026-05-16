import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
    # Verificacao de integridade estatistica
    resumo = df.describe()
    num_missing = df.isnull().sum()
    
    # Analise de Correlacao Espacial e Multicolinearidade Inicial
    corr_matrix = df.corr()
    
    # Visualizacao de Outliers e Distribuicao
    sns.pairplot(df, diag_kind='kde')
    plt.show()
    
    return resumo, num_missing
