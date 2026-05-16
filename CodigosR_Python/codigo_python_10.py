import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

def analise_time_series(data):
    # 1. Teste de Estacionariedade
    adf_result = sm.tsa.stattools.adfuller(data)
    print(f''ADF Statistic: {adf_result[0]}, p-value: {adf_result[1]}'')
    
    # 2. Identificacao ACF/PACF
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sm.graphics.tsa.plot_acf(data, ax=axes[0])
    sm.graphics.tsa.plot_pacf(data, ax=axes[1])
    plt.show()
    
    # 3. Estimacao SARIMAX
    model = sm.tsa.SARIMAX(data, order=(1, 1, 1))
    results = model.fit()
    print(results.summary())
    
    # 4. Previsao
    forecast = results.get_forecast(steps=12)
    forecast.summary_frame().plot()
