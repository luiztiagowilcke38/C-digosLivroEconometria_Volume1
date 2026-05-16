library(urca)
library(vars)

# Simulacao similar
set.seed(42)
n <- 500
trend <- cumsum(rnorm(n))
y1 <- 0.8 * trend + rnorm(n, 0, 0.5)
y2 <- 1.2 * trend + rnorm(n, 0, 0.5)
data <- data.frame(y1, y2)

# Teste de Johansen (ca.jo)
johansen_test <- ca.jo(data, type="trace", K=2, spec="transitory")
summary(johansen_test)

# Conversao para VECM/VAR (vec2var)
vecm_model <- vec2var(johansen_test, r=1)
# Previsao de 5 passos a frente
predict(vecm_model, n.ahead=5)
