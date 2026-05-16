library(foreach)
library(doParallel)
library(MASS)

# Funcao de Log-Verossimilhanca Probit
log_like_probit <- function(beta, y, X) {
  eta <- as.vector(X %*% beta)
  # Usando pnorm com log=TRUE para estabilidade numerica
  ll_1 <- sum(pnorm(eta[y == 1], log.p = TRUE))
  ll_0 <- sum(pnorm(-eta[y == 0], log.p = TRUE))
  return(ll_1 + ll_0)
}

# Distribuicao Posterior Nao Normalizada (log)
log_posterior <- function(beta, y, X, prior_mean, prior_var_inv) {
  ll <- log_like_probit(beta, y, X)
  prior <- -0.5 * t(beta - prior_mean) %*% prior_var_inv %*% (beta - prior_mean)
  return(ll + prior)
}

# Algoritmo de Metropolis-Hastings (Random Walk)
probit_mh_sampler <- function(y, X, iter=10000, burnin=2000, tune=0.05) {
  k <- ncol(X)
  samples <- matrix(0, nrow=iter, ncol=k)
  
  # Inicializacao via MLE (glm)
  init_fit <- glm(y ~ X - 1, family = binomial(link=''probit''))
  current_beta <- coef(init_fit)
  prop_cov <- vcov(init_fit) * tune # Matriz de covariancia do proposal
  
  prior_mean <- rep(0, k)
  prior_var_inv <- diag(1/100, k)
  
  current_log_post <- log_posterior(current_beta, y, X, prior_mean, prior_var_inv)
  aceitacoes <- 0
  
  for(i in 1:iter) {
    # Proposta de Salto (Random Walk com Normal Multivariada)
    prop_beta <- mvrnorm(1, mu=current_beta, Sigma=prop_cov)
    prop_log_post <- log_posterior(prop_beta, y, X, prior_mean, prior_var_inv)
    
    # Razao de Aceitacao log(alpha)
    log_alpha <- prop_log_post - current_log_post
    
    if(log(runif(1)) < log_alpha) {
      current_beta <- prop_beta
      current_log_post <- prop_log_post
      aceitacoes <- aceitacoes + 1
    }
    samples[i, ] <- current_beta
  }
  
  cat(''Taxa de Aceitacao: '', aceitacoes / iter, ''\n'')
  return(samples[(burnin+1):iter, ])
}

# Integrando com Monte Carlo Paralelo para Estudo de Frequencia
setup_parallel_mc <- function(R=100, N=500, n_cores=4) {
  cl <- makeCluster(n_cores)
  registerDoParallel(cl)
  
  resultados <- foreach(r = 1:R, .combine=rbind, .packages=c(''MASS'')) %dopar% {
    set.seed(r + 1000)
    X_sim <- cbind(1, rnorm(N), runif(N))
    beta_true <- c(-0.5, 1.2, -0.8)
    
    y_star <- X_sim %*% beta_true + rnorm(N)
    y_sim <- ifelse(y_star > 0, 1, 0)
    
    post_samples <- probit_mh_sampler(y_sim, X_sim, iter=5000, burnin=1000, tune=1.5)
    
    # Retorna as medias a posteriori
    colMeans(post_samples)
  }
  stopCluster(cl)
  
  bias <- colMeans(resultados) - c(-0.5, 1.2, -0.8)
  print(''Vies Frequencista da Media Posterior Bayesiana:'')
  print(bias)
  return(resultados)
}

# sim_res <- setup_parallel_mc(R=100, N=500)
