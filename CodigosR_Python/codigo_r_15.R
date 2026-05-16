set.seed(42)
n <- 500

# Gerando regressores e Y
X1 <- cbind(1, matrix(rnorm(n*2), ncol=2))   # k1 = 3 (incluindo intercepto)
X2 <- matrix(rnorm(n*2), ncol=2)              # k2 = 2
beta1_true <- c(1.0, -0.5, 0.8)
beta2_true <- c(2.0, -1.3)
y <- X1 %*% beta1_true + X2 %*% beta2_true + rnorm(n)

# === METODO 1: MQO Completo ===
X_full <- cbind(X1, X2)
beta_full <- solve(t(X_full) %*% X_full, t(X_full) %*% y)
beta2_full <- beta_full[4:5]
cat(''MQO Completo (beta2):'', beta2_full, ''\n'')

# === METODO 2: Aniquilador M1 explícito ===
P1 <- X1 %*% solve(t(X1) %*% X1) %*% t(X1)  # Projetor
M1 <- diag(n) - P1                            # Aniquilador

# Checar propriedades
cat(''\nPropriedades de M1:\n'')
cat(''  Idempotente M1^2=M1?'', isTRUE(all.equal(M1 %*% M1, M1)), ''\n'')
cat(''  Simetrico?          '', isTRUE(all.equal(M1, t(M1))), ''\n'')
cat(''  max|M1 X1|:         '', max(abs(M1 %*% X1)), ''\n'')

# Parcializacao e regressao FWL
y_tilde  <- M1 %*% y
X2_tilde <- M1 %*% X2
beta2_fwl <- solve(t(X2_tilde) %*% X2_tilde, t(X2_tilde) %*% y_tilde)
cat(''\nFWL (beta2):'', beta2_fwl, ''\n'')
cat(''Diferenca maxima:'', max(abs(beta2_full - beta2_fwl)), ''\n'')

# === METODO 3: Via lm() - Forma pratica do dia-a-dia ===
df <- data.frame(y=y, x1a=X1[,2], x1b=X1[,3], x2a=X2[,1], x2b=X2[,2])

# Residuos das regressoes auxiliares
y_res   <- residuals(lm(y ~ x1a + x1b, data=df))
x2a_res <- residuals(lm(x2a ~ x1a + x1b, data=df))
x2b_res <- residuals(lm(x2b ~ x1a + x1b, data=df))

# Regressao FWL final
mod_fwl <- lm(y_res ~ x2a_res + x2b_res - 1)  # sem intercepto (ja parcializado)
cat(''\nFWL via lm():'', coef(mod_fwl), ''\n'')

# === CORRELACOES PARCIAIS (aplicacao direta do FWL) ===
# A correlacao parcial de X2a com Y, controlando X1 e X2b, e':
cat(''\nCorrelacao parcial de X2a com Y (ctrl X1, X2b):\n'')
cat(''  r ='', cor(residuals(lm(y ~ x1a+x1b+x2b, data=df)),
                 residuals(lm(x2a ~ x1a+x1b+x2b, data=df))), ''\n'')

# === R2 parcial: contribuicao adicional de X2 dado X1 ===
rss_full    <- sum(residuals(lm(y ~ x1a+x1b+x2a+x2b, data=df))^2)
rss_reduced <- sum(residuals(lm(y ~ x1a+x1b,         data=df))^2)
r2_partial  <- (rss_reduced - rss_full) / rss_reduced
cat(sprintf(''\nR2 Parcial de X2 (contribuicao alem de X1): %.4f\n'', r2_partial))

# === APLICACAO ECONOMICA: Efeito de Educacao controlando Experiencia ===
set.seed(1)
n2 <- 300
experiencia <- runif(n2, 0, 30)      # X1: controle
educacao    <- 0.3 * experiencia + rnorm(n2, 12, 3)  # X2: interesse (correlac. com X1!)
salario     <- 500 + 50*educacao + 20*experiencia + rnorm(n2, 0, 200)
df2 <- data.frame(salario, educacao, experiencia)

# Sem FWL: Efeito ingênuo de educacao (confundido com experiencia)
cat(''\nEfeito ingênuo de educacao:'', coef(lm(salario~educacao, data=df2))[2], ''\n'')

# Com FWL: Efeito parcializado (removendo o canal experiencia)
ed_res  <- residuals(lm(educacao ~ experiencia, data=df2))
sal_res <- residuals(lm(salario  ~ experiencia, data=df2))
cat(''Efeito FWL de educacao:  '', coef(lm(sal_res ~ ed_res))[2], ''\n'')

# Confirmar com MQO completo
cat(''MQO completo (educacao): '', coef(lm(salario~educacao+experiencia, data=df2))[2], ''\n'')
