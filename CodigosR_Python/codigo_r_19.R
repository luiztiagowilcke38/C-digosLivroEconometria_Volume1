library(glmnet)

set.seed(42)
n <- 200; k <- 4
X   <- cbind(1, matrix(rnorm(n*(k-1)), ncol=k-1))
beta_t <- c(1.0, 2.0, -1.5, 0.8)
y   <- X %*% beta_t + rnorm(n, 0, 1.5)

# =========================================================
# PARTE A: Ridge via formula KKT
# =========================================================
ridge_kkt_r <- function(X, y, lam){
    k <- ncol(X)
    solve(t(X) %*% X + lam * diag(k), t(X) %*% y)
}

cat(''--- Ridge via KKT ---\n'')
for(lam in c(0, 0.1, 1.0, 10.0)){
    b <- ridge_kkt_r(X, y, lam)
    cat(sprintf(''  lambda=%5.1f: beta=%s  ||e||^2=%.2f\n'',
                lam, paste(round(b,3), collapse='' ''),
                sum((y - X %*% b)^2)))
}

# =========================================================
# PARTE B: LASSO - Verificacao da Condicao KKT
# =========================================================
cat(''\n--- Verificacao KKT do LASSO ---\n'')
lam_lasso  <- 0.5
fit_lasso  <- glmnet(X[,-1], y, alpha=1, lambda=lam_lasso/n,
                     intercept=TRUE, standardize=FALSE)
b_lasso    <- c(coef(fit_lasso)[1], as.vector(coef(fit_lasso)[-1]))
b_lasso_full <- c(b_lasso[1], b_lasso[-1])

resid_l  <- y - X %*% b_lasso_full
grad_ll  <- t(X) %*% resid_l

for(j in 1:k){
    if(abs(b_lasso_full[j]) > 1e-6){
        err <- abs(abs(grad_ll[j]) - lam_lasso)
        cat(sprintf(''  beta[%d]=%7.4f != 0: |grad_j - lam|=%.2e (deve~0)\n'',
                    j, b_lasso_full[j], err))
    } else {
        ok <- abs(grad_ll[j]) <= lam_lasso + 1e-6
        cat(sprintf(''  beta[%d]=%7.4f =  0: |grad_j|=%.4f <= %.1f ? %s\n'',
                    j, b_lasso_full[j], abs(grad_ll[j]), lam_lasso, ok))
    }
}

# =========================================================
# PARTE C: MQO restrito - KKT com restricao de igualdade R*b = r
# =========================================================
cat(''\n--- MQO Restrito com R*beta=r (KKT Igualdade) ---\n'')
R_mat   <- matrix(c(0,0,1,1), nrow=1)          # beta_2 + beta_3 = 0
r_vec   <- 0

XtX_inv <- solve(t(X) %*% X)
b_ols   <- XtX_inv %*% t(X) %*% y
RXR     <- R_mat %*% XtX_inv %*% t(R_mat)

lambda_star <- solve(RXR, R_mat %*% b_ols - r_vec)
b_rest      <- b_ols - XtX_inv %*% t(R_mat) %*% lambda_star

cat(sprintf(''beta OLS irrestrito: %s\n'', paste(round(b_ols,4), collapse='' '')))
cat(sprintf(''beta KKT restrito:   %s\n'', paste(round(b_rest,4), collapse='' '')))
cat(sprintf(''Restricao R*b=%.0f:   %.6f\n'', r_vec, R_mat %*% b_rest))
cat(sprintf(''Multiplicador lambda*= %.4f\n'', lambda_star[1]))

# Teste estatistico: H0: R*beta = r via estatistica F
b_ols_v <- as.vector(b_ols)
b_rest_v<- as.vector(b_rest)
SSE_rest <- sum((y - X %*% b_rest_v)^2)
SSE_ols  <- sum((y - X %*% b_ols_v)^2)
q <- nrow(R_mat)
F_stat <- ((SSE_rest - SSE_ols)/q) / (SSE_ols/(n - k))
p_val  <- 1 - pf(F_stat, q, n-k)
cat(sprintf(''Teste F da restricao: F=%.4f, p=%.4f  %s\n'',
            F_stat, p_val, ifelse(p_val > 0.05, ''Nao rejeita H0'', ''Rejeita H0'')))

# =========================================================
# PARTE D: Condicoes de 2a ordem - Hessiana na restricao ativa
# =========================================================
cat(''\n--- Condicoes de 2a Ordem (Hessiana Restrita) ---\n'')
# Para MQO restrito, a Hessiana do Lagrangeano e' 2*X'X (quadratica)
# Deve ser positiva definida no espaco nulo de R (minimizacao)
# Para maximizacao de log-vero, a Hessiana deve ser negativa definida
Hess <- -2 * t(X) %*% X   # Hessiana do criterio de maximizacao (-||e||^2)

# Espaco nulo de R: vetores d tal que R*d = 0
N_null <- MASS::Null(t(R_mat))   # Colunas formam base do nucleo de R

# Hessiana restrita ao espaco nulo
H_rest <- t(N_null) %*% Hess %*% N_null
eigs   <- eigen(H_rest)$values
cat(sprintf(''Autovalores da Hessiana restrita: %s\n'', paste(round(eigs,3), collapse='' '')))
cat(sprintf(''Todos negativos? %s => Condicao 2a ordem satisfeita (maximo)\n'',
            all(eigs < 0)))
