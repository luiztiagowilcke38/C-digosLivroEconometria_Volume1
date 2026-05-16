library(MASS)        # mvrnorm
library(mvtnorm)     # dmvnorm, pmvnorm

set.seed(42)
k  <- 4
mu <- c(1.0, -0.5, 2.0, 0.3)

# Gerar Sigma positiva definida
A     <- matrix(rnorm(k*k), k, k)
Sigma <- (A %*% t(A)) / k + diag(k)
cat(''Autovalores de Sigma (todos > 0):'', round(eigen(Sigma)$values, 3), ''\n'')

# =========================================================
# PARTE A: Densidade e Simulacao
# =========================================================
x0 <- rep(0, k)

# Densidade via dmvnorm (mvtnorm)
dens_mvn <- dmvnorm(x0, mean=mu, sigma=Sigma)
cat(sprintf(''Densidade em x=0: %.6f\n'', dens_mvn))

# Calculo manual
diff    <- x0 - mu
inv_S   <- solve(Sigma)
det_S   <- det(Sigma)
mahal2  <- as.numeric(t(diff) %*% inv_S %*% diff)
f_man   <- (1 / ((2*pi)^(k/2) * sqrt(det_S))) * exp(-0.5 * mahal2)
cat(sprintf(''Densidade manual:  %.6f\n'', f_man))
cat(sprintf(''Distancia Mahalanobis: %.4f\n'', sqrt(mahal2)))

# Simulacao via Decomposicao de Cholesky (base do mvrnorm)
N   <- 50000
L   <- t(chol(Sigma))          # chol() retorna upper; transpose para lower
Z   <- matrix(rnorm(k*N), k, N)
X_s <- mu + L %*% Z            # k x N

cat(''\nMedia amostral:'', round(rowMeans(X_s), 3))
cat(''\nMax dif Sigma estimada vs teorica:'', round(max(abs(cov(t(X_s)) - Sigma)), 4), ''\n'')

# =========================================================
# PARTE B: Teorema de Cramer-Wold
# =========================================================
cat(''\n--- Verificacao Cramer-Wold (Shapiro-Wilk em combinacoes lineares) ---\n'')
for(i in 1:5){
    cc     <- rnorm(k); cc <- cc/sqrt(sum(cc^2))  # vetor aleatorio normalizado
    proj   <- as.vector(cc %*% X_s)   # N escalares
    mu_p   <- as.numeric(cc %*% mu)
    sig_p  <- sqrt(as.numeric(t(cc) %*% Sigma %*% cc))
    sw     <- shapiro.test(sample(proj, 5000))
    cat(sprintf(''  c%d: E=%.3f(%.3f) Std=%.3f(%.3f) Shapiro p=%.3f %s\n'',
                i, mu_p, mean(proj), sig_p, sd(proj), sw$p.value,
                ifelse(sw$p.value > 0.05, ''OK-Normal'', ''Atencao!'')))
}

# =========================================================
# PARTE C: Condicional da Normal Multivariada
# =========================================================
# Particionando X = [X1 (1:2), X2 (3:4)]
mu1 <- mu[1:2]; mu2 <- mu[3:4]
S11 <- Sigma[1:2, 1:2]; S12 <- Sigma[1:2, 3:4]
S21 <- Sigma[3:4, 1:2]; S22 <- Sigma[3:4, 3:4]

x2_obs  <- c(1.5, 0.0)
mu_cond <- mu1 + S12 %*% solve(S22) %*% (x2_obs - mu2)
S_cond  <- S11 - S12 %*% solve(S22) %*% S21

cat(''\n--- Condicional X1 | X2 ---\n'')
cat(''Media condicional (teorica): '', round(mu_cond, 4), ''\n'')
cat(''Covariancia condicional:\n''); print(round(S_cond, 4))

# Verificacao empirica via filtro de amostras proximas a x2_obs
tol   <- 0.15
idx   <- which(abs(X_s[3,] - x2_obs[1]) < tol & abs(X_s[4,] - x2_obs[2]) < tol)
cat(sprintf(''Media condicional (amostral, n=%d): %.4f  %.4f\n'',
            length(idx), mean(X_s[1, idx]), mean(X_s[2, idx])))

# =========================================================
# PARTE D: Forma Quadratica chi2(k) 
# =========================================================
# (X - mu)' Sigma^{-1} (X - mu) ~ chi2(k)
X_t      <- t(X_s)         # N x k
diffs    <- sweep(X_t, 2, mu)
mahal_sq <- rowSums((diffs %*% inv_S) * diffs)

cat(sprintf(''\nMahalanobis^2 ~ chi2(%d): media=%.3f (teorico=%d)\n'', k, mean(mahal_sq), k))
ks_p <- ks.test(mahal_sq, ''pchisq'', df=k)$p.value
cat(sprintf(''KS p-valor: %.4f  %s\n'', ks_p, ifelse(ks_p>0.05, ''Aceita chi2'', ''Rejeita'')))

# =========================================================
# PARTE E: Aplicacao em Econometria - Teste Conjunto de Hipoteses
# =========================================================
# R*beta = r via estatistica Wald ~ chi2(q) quando n grande
# F = (R*beta_hat - r)' [R (X'X)^-1 R']^-1 (R*beta_hat - r) / (q * sigma^2)
set.seed(55)
n_reg <- 500
X_reg <- cbind(1, matrix(rnorm(n_reg*3), ncol=3))
beta_reg <- c(1, 2, -1, 0.5)
y_reg <- X_reg %*% beta_reg + rnorm(n_reg)
mod <- lm(y_reg ~ X_reg[,-1])

# Teste H0: beta2 = 2 AND beta3 = -1 (q=2 restricoes)
R_mat <- matrix(c(0,1,0,0, 0,0,1,0), nrow=2, byrow=TRUE)
r_vec <- c(2, -1)
b_hat <- coef(mod)
V_hat <- vcov(mod)

Rb_r    <- R_mat %*% b_hat - r_vec
RVR_inv <- solve(R_mat %*% V_hat %*% t(R_mat))
W_stat  <- as.numeric(t(Rb_r) %*% RVR_inv %*% Rb_r)  # chi2(2) assintoticamente
p_wald  <- 1 - pchisq(W_stat, df=2)
cat(sprintf(''\nTeste Wald H0: beta2=2, beta3=-1\n''))
cat(sprintf(''W = %.4f ~ chi2(2), p-valor = %.4f  %s\n'',
            W_stat, p_wald, ifelse(p_wald>0.05, ''Nao rejeita H0'', ''Rejeita H0'')))
