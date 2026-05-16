library(gmm)
set.seed(123)
n <- 500
r <- rnorm(n, 0.05, 0.01) # Retorno real
dc <- rnorm(n, 0.02, 0.005) # Crescimento do consumo

# Condicao de momento: E[beta * (1+r) * (c1/c0)^-gamma - 1] = 0
g_moments <- function(theta, x) {
  beta <- theta[1]
  gamma <- theta[2]
  r <- x[,1]
  dc <- x[,2]
  m1 <- beta * (1 + r) * exp(-gamma * dc) - 1
  return(cbind(m1, m1 * r)) # Instrumento: retorno defasado (proxy)
}

x <- cbind(r, dc)
fit_gmm <- gmm(g_moments, x, t0 = c(beta = 0.95, gamma = 2))
summary(fit_gmm)
