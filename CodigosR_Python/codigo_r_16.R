library(survival)  # Para censura; ou usar pacote censReg / AER

set.seed(42)
n <- 5000

# =========================================================
# PARTE A: Verificacao Numerica das Integrais
# =========================================================
mu <- 3.0; sigma <- 1.5

# Integral de Riemann via integrate()
f_integrand <- function(x) x * dnorm(x, mu, sigma)
res_riemann  <- integrate(f_integrand, lower=-Inf, upper=Inf)
cat(sprintf(''E[X] (Riemann/integrate): %.6f\n'', res_riemann$value))

# Aproximacao de Lebesgue (funcoes simples tipo escada)
x_grid   <- seq(mu - 6*sigma, mu + 6*sigma, length.out=10000)
delta_x  <- x_grid[2] - x_grid[1]
E_lebesgue <- sum(x_grid * dnorm(x_grid, mu, sigma) * delta_x)
cat(sprintf(''E[X] (Lebesgue discreto): %.6f\n'', E_lebesgue))

# Monte Carlo (lei dos grandes numeros)
cat(sprintf(''E[X] (Monte Carlo 500k):   %.6f\n'', mean(rnorm(500000, mu, sigma))))

# =========================================================
# PARTE B: Esperanca de Variavel Mista (Tobit)
# =========================================================
beta_tobit <- c(1.5, 2.0)  # intercepto e slope
x_cov      <- rnorm(n)
sig_tobit  <- 1.2
Y_star <- beta_tobit[1] + beta_tobit[2]*x_cov + rnorm(n, 0, sig_tobit)
Y_obs  <- pmax(0, Y_star)

cat(sprintf(''\nProporção censurada: %.3f\n'', mean(Y_obs == 0)))

# Formula analitica E[Y_obs] = Phi(XB/sig)*(XB) + sig*phi(XB/sig)
xb     <- beta_tobit[1] + beta_tobit[2]*x_cov
E_form <- pnorm(xb/sig_tobit) * xb + sig_tobit * dnorm(xb/sig_tobit)
cat(sprintf(''E[Y_obs] formula Tobit:   %.4f\n'', mean(E_form)))
cat(sprintf(''E[Y_obs] amostral:         %.4f\n'', mean(Y_obs)))

# =========================================================
# PARTE C: Estimacao Tobit por MLE (simulando censReg)
# =========================================================
# Log-verossimilhanca do Tobit manualmente
tobit_loglik <- function(params, y, x){
    b0 <- params[1]; b1 <- params[2]; sig <- exp(params[3])  # sig>0 via exp()
    xb <- b0 + b1*x
    # Observacoes nao-censuradas: contribuicao = log(phi(y-xb)/sig)
    # Observacoes censuradas em 0: contribuicao = log(Phi(-xb/sig))
    censored <- (y == 0)
    ll <- sum(dnorm(y[!censored], mean=xb[!censored], sd=sig, log=TRUE)) +
          sum(pnorm(-xb[censored]/sig, log.p=TRUE))
    return(-ll)  # Negativo para minimizar
}

opt <- optim(c(1.0, 1.5, log(1.0)), tobit_loglik,
             y=Y_obs, x=x_cov, method=''BFGS'')
cat(sprintf(''\nMLE Tobit manual: b0=%.3f, b1=%.3f, sigma=%.3f\n'',
            opt$par[1], opt$par[2], exp(opt$par[3])))
cat(sprintf(''Verdadeiros:      b0=%.3f, b1=%.3f, sigma=%.3f\n'',
            beta_tobit[1], beta_tobit[2], sig_tobit))

# =========================================================
# PARTE D: Teorema da Convergencia Dominada (DCT) Numericamente
# =========================================================
# DCT: Se |f_n| <= g integravel, entao lim integral f_n = integral lim f_n
# Demonstramos com f_n(x) = (1 + x/n)^n * I(x in [0,5]) -> exp(x) * I(x in [0,5])
cat(''\nDCT - Convergencia de integral:\n'')
x_dom <- seq(0, 5, length.out=10000); dx <- x_dom[2]-x_dom[1]
E_limite <- sum(exp(x_dom) * dx)     # lim f_n = e^x
for(N in c(2, 10, 50, 500)){
    f_n  <- (1 + x_dom/N)^N
    E_n  <- sum(f_n * dx)
    cat(sprintf(''  n=%4d: integral f_n = %.5f (limite = %.5f)\n'', N, E_n, E_limite))
}
