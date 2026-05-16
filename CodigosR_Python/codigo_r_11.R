set.seed(123)
n <- 1000
U <- rnorm(n)
Z <- rnorm(n)
X_endo <- 0.7*Z + 0.5*U + rnorm(n, 0, 0.5)
Y <- 2.0*X_endo + 3.0*U + rnorm(n)

X_mat <- cbind(1, X_endo)
Z_mat <- cbind(1, Z)

# 1o Estagio
beta_1st <- solve(t(Z_mat) %*% Z_mat, t(Z_mat) %*% X_mat)
X_hat    <- Z_mat %*% beta_1st

# 2o Estagio
beta_iv <- solve(t(X_hat) %*% X_mat, t(X_hat) %*% Y)
cat(sprintf(''2SLS: alpha=%.3f, beta=%.3f\n'', beta_iv[1], beta_iv[2]))

# Residuos do 1o estagio para teste de Hausman
resid_1st <- X_endo - X_hat[,2]

# TESTE DE HAUSMAN: Incluir residuos_1o no reg principal
# Se significativo => endogeneidade detectada (rejeita H0: MQO consistente)
mod_hausman <- lm(Y ~ X_endo + resid_1st)
cat(''\nTeste de Hausman (coef do resid 1o estagio):\n'')
print(summary(mod_hausman)$coefficients[3,])

# Versao automatica via AER
library(AER)
mod_iv <- ivreg(Y ~ X_endo | Z)
print(summary(mod_iv, diagnostics=TRUE))
