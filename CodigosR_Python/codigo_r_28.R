library(rugarch)
library(quantmod)

# Obter dados financeiros (Apple AAPL)
getSymbols(''AAPL'', from=''2020-01-01'', to=''2023-12-31'')
ret <- dailyReturn(Cl(AAPL), type='log')

# Especificacao do modelo ARMA(1,1)-eGARCH(1,1) com choques t-Student
spec_egarch <- ugarchspec(
  variance.model = list(model = ''eGARCH'', garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(1, 1), include.mean = TRUE),
  distribution.model = ''std'' # Captura caudas gordas
)

# Estimar e Prever 10 passos
fit <- ugarchfit(spec = spec_egarch, data = ret)
show(fit) # Imprime pseudo-t e Ljung-Box Test

forecasts <- ugarchforecast(fit, n.ahead = 10)
plot(forecasts, which = 1)
