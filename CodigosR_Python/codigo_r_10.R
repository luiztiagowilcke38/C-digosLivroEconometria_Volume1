# Reproducao identica em R (mesmo seed de geracao)
set.seed(42)
n <- 500; k <- 3
X_raw <- matrix(rnorm(n*k), ncol=k)
X <- cbind(1, X_raw)  # Adiciona intercepto
beta_true <- c(2.0, 1.5, -0.8, 0.5)
y <- X %*% beta_true + rnorm(n, 0, 1.5)

# SOLUCAO 1: MQO pela formula fechada
beta_ols <- solve(t(X) %*% X, t(X) %*% y)
cat(''MQO (formula fechada):'', beta_ols, ''\n'')

# SOLUCAO 2: QR nativa do R (usada internamente pelo lm())
qr_decomp <- qr(X)
beta_qr   <- qr.coef(qr_decomp, y)
cat(''MQO (QR nativo R):   '', beta_qr, ''\n'')

# SOLUCAO 3: via funcao lm() padrao
mod  <- lm(y ~ X_raw)
cat(''MQO (lm):            '', coef(mod), ''\n'')

# Calculo manual das matrizes diagnosticas
H <- X %*% solve(t(X) %*% X) %*% t(X)      # Matriz Hat
residuos <- y - X %*% beta_ols
sigma2   <- sum(residuos^2) / (n - k - 1)

# Residuos studentizados internamente
leverage   <- diag(H)
se_res     <- sqrt(sigma2 * (1 - leverage))
stud_res   <- residuos / se_res

cat(sprintf(''\nMax Leverage %.4f | Outliers (|r|>3): %d\n'',
            max(leverage), sum(abs(stud_res) > 3)))
