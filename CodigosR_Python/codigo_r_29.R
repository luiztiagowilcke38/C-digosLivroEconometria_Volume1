library(rmgarch)
library(zoo)

# Definir especificacao GARCH univariada para cada ativo
uspec <- ugarchspec(variance.model = list(model='sGARCH', garchOrder=c(1,1)),
                    mean.model = list(armaOrder=c(0,0), include.mean=FALSE),
                    distribution.model = 'std')

# Replica para dois ativos
spec_list <- multispec(replicate(2, uspec))

# Especificacao DCC
dcc_spec <- dccspec(uspec = spec_list, dccOrder = c(1, 1),
                    distribution = 'mvt')

# Dados sinteticos de dois ativos correlacionados
set.seed(10)
R_matrix <- matrix(c(1, 0.6, 0.6, 1), nrow=2)
L <- t(chol(R_matrix))
raw_data <- matrix(rt(1000*2, df=5), ncol=2) %*% t(L)

# Estimar DCC
dcc_fit <- dccfit(dcc_spec, data=raw_data)
print(dcc_fit)

# Extrair correlacoes condicionais ao longo do tempo
rho_t <- rcor(dcc_fit) # Array 2x2xT
rho_12 <- rho_t[1,2,]
plot(rho_12, type='l', main='Correlacao Condicional Dinamica DCC', ylab='rho_t')
