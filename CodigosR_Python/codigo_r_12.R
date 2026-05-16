set.seed(0)
n <- 2000
omega_t <- 0.01; alpha_t <- 0.10; beta_t <- 0.85

# Simular GARCH(1,1) manualmente
sigma2 <- rep(omega_t/(1-alpha_t-beta_t), n)
eps <- rep(0, n)
for(t in 2:n){
    sigma2[t] <- omega_t + alpha_t * eps[t-1]^2 + beta_t * sigma2[t-1]
    eps[t]    <- rnorm(1, 0, sqrt(sigma2[t]))
}
ret <- eps

# Log-Verossimilhanca GARCH
garch_ll <- function(params, r){
    omega <- params[1]; alpha <- params[2]; beta <- params[3]
    if(omega<=0 || alpha<0 || beta<0 || alpha+beta>=1) return(1e10)
    T <- length(r)
    sig2 <- rep(var(r), T)
    for(t in 2:T) sig2[t] <- omega + alpha*r[t-1]^2 + beta*sig2[t-1]
    ll <- -0.5 * sum(log(2*pi) + log(sig2) + r^2/sig2)
    return(-ll)  # negativo para minimizar
}

# Otimizacao por L-BFGS-B com bounds positivos
fit <- optim(c(0.02, 0.08, 0.88), garch_ll, r=ret,
             method=''L-BFGS-B'',
             lower=c(1e-6, 1e-6, 1e-6),
             upper=c(1, 0.5, 0.999))
cat(sprintf(''omega=%.5f alpha=%.5f beta=%.5f\n'', fit$par[1], fit$par[2], fit$par[3]))

# Diagnostico: Residuos padronizados devem ser iid N(0,1)
omega_h <- fit$par[1]; alpha_h <- fit$par[2]; beta_h <- fit$par[3]
sig2_h <- rep(var(ret), n)
for(t in 2:n) sig2_h[t] <- omega_h + alpha_h*ret[t-1]^2 + beta_h*sig2_h[t-1]
resid_std <- ret / sqrt(sig2_h)
cat(sprintf(''Kurtosis dos residuos padronizados: %.3f (deve ser ~3 se Normal)\n'',
            mean(resid_std^4) / mean(resid_std^2)^2))
