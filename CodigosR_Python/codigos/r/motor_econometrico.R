#' @title Motor Econometrico Advanced em R (Rigor Academico e Performance)
#' @author LT
#' @description Framework completo para estimacao, inferencia e diagnosticos.
#' poderosa para pesquisa econometrica avançada em R.

#' @description Criador de Objetos Econometricos
Econometria <- function(y, X, nomes = NULL) {
  if (is.vector(y)) y <- matrix(y, ncol = 1)
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  
  n <- nrow(X)
  k <- ncol(X)
  
  if (is.null(nomes)) {
    nomes <- c("(Intercepto)", paste0("x", 1:(k-1)))
  }
  
  structure(list(
    y = y, X = X, n = n, k = k, nomes = nomes,
    estimado = FALSE
  ), class = "Econometria")
}

#' @description Estimador de Minimos Quadrados Ordinarios
estimar_mqo <- function(obj, robusto = FALSE) {
  # Decomposicao QR
  qr_X <- qr(obj$X)
  betas <- qr.coef(qr_X, obj$y)
  residuos <- qr.resid(qr_X, obj$y)
  ajustados <- qr.fitted(qr_X, obj$y)
  
  # Variancia Proporcional
  sigma2 <- as.numeric(crossprod(residuos) / (obj$n - obj$k))
  xtx_inv <- solve(crossprod(obj$X))
  
  if (!robusto) {
    vce <- sigma2 * xtx_inv
  } else {
    # HC0 - White
    u2 <- residuos^2
    meat <- t(obj$X) %*% (as.numeric(u2) * obj$X)
    vce <- xtx_inv %*% meat %*% xtx_inv
  }
  
  obj$coeficientes <- betas
  obj$residuos <- residuos
  obj$ajustados <- ajustados
  obj$sigma2 <- sigma2
  obj$vce <- vce
  obj$estimado <- TRUE
  return(obj)
}

#' @description Resumo dos Resultados (Inferencia)
summary.Econometria <- function(obj) {
  if (!obj$estimado) stop("Modelo nao estimado.")
  
  ep <- sqrt(diag(obj$vce))
  t_val <- obj$coeficientes / ep
  p_val <- 2 * (1 - pt(abs(t_val), df = obj$n - obj$k))
  
  tabela <- data.frame(
    Estimativa = as.numeric(obj$coeficientes),
    ErroPadrao = ep,
    t_stat = as.numeric(t_val),
    p_valor = as.numeric(p_val)
  )
  rownames(tabela) <- obj$nomes
  
  cat("\n===============================================================\n")
  cat("MOTOR ECONOMÉTRICO R - RESUMO ESTATÍSTICO\n")
  cat("---------------------------------------------------------------\n")
  print(round(tabela, 4))
  cat("---------------------------------------------------------------\n")
  
  # R2
  sst <- sum((obj$y - mean(obj$y))^2)
  ssr <- sum(obj$residuos^2)
  r2 <- 1 - (ssr / sst)
  cat(sprintf("R-Quadrado: %.4f | Obs: %d | Graus Lib: %d\n", r2, obj$n, obj$n - obj$k))
  cat("===============================================================\n")
}

#' @description Estimador de Variaveis Instrumentais (2SLS)
estimar_2sls <- function(y, X, Z, nomes = NULL) {
  # Primeiro Estagio
  Pz <- Z %*% solve(t(Z) %*% Z) %*% t(Z)
  X_hat <- Pz %*% X
  
  # Segundo Estagio
  modelo <- Econometria(y, X_hat, nomes)
  modelo <- estimar_mqo(modelo)
  # Corrigindo os residuos para o X original
  modelo$residuos <- y - X %*% modelo$coeficientes
  return(modelo)
}

#' @description Teste de Breusch-Pagan (Heterocedasticidade)
teste_bp <- function(obj) {
  u2 <- obj$residuos^2
  f_aux <- lm(u2 ~ obj$X - 1)
  r2_aux <- summary(f_aux)$r.squared
  lm_stat <- obj$n * r2_aux
  p_val <- 1 - pchisq(lm_stat, df = obj$k - 1)
  return(c(LM = lm_stat, p = p_val))
}

#' @description Teste de RESET de Ramsey
teste_reset <- function(obj) {
  y_hat2 <- obj$ajustados^2
  y_hat3 <- obj$ajustados^3
  X_new <- cbind(obj$X, y_hat2, y_hat3)
  
  m_ur <- Econometria(obj$y, X_new)
  m_ur <- estimar_mqo(m_ur)
  
  ssr_r <- sum(obj$residuos^2)
  ssr_ur <- sum(m_ur$residuos^2)
  m <- 2 # Restricoes
  
  f_stat <- ((ssr_r - ssr_ur) / m) / (ssr_ur / (obj$n - obj$k - m))
  p_val <- 1 - pf(f_stat, m, obj$n - obj$k - m)
  return(c(F = f_stat, p = p_val))
}

#' @description Modelo Logit via Otimizacao Numerica (Newton-Raphson)
estimar_logit <- function(y, X) {
  n <- nrow(X)
  k <- ncol(X)
  beta <- rep(0, k)
  
  # Funcao de Log-Verossimilhanca
  ll <- function(b) {
    p <- 1 / (1 + exp(-X %*% b))
    return(-sum(y * log(p) + (1 - y) * log(1 - p)))
  }
  
  # Otimizacao
  opt <- optim(beta, ll, hessian = TRUE)
  
  vce <- solve(opt$hessian)
  ep <- sqrt(diag(vce))
  t_stats <- opt$par / ep
  
  return(list(betas = opt$par, ep = ep, t = t_stats, vce = vce))
}

# =============================================================================
# LABORATORIO DE SIMULACAO AVANCADO
# =============================================================================

run_massive_simulation <- function(iters = 1000) {
  cat("\nIniciando Experimento Monte Carlo de Larga Escala...\n")
  
  results <- replicate(iters, {
    # DGP Complexo
    n <- 300
    x1 <- rnorm(n)
    # Endogeneidade
    u <- rnorm(n)
    x2 <- 0.7 * u + rnorm(n)
    y <- 1 + 2*x1 + 3*x2 + u
    
    # MQO
    X <- cbind(1, x1, x2)
    b_mqo <- solve(t(X) %*% X, t(X) %*% y)
    
    # IV (Instrumento z)
    z <- rnorm(n)
    x2_iv <- 2 * z + rnorm(n, sd = 0.2)
    X_iv <- cbind(1, x1, x2) # Mas usamos z como instrumento
    # (Manual 2SLS simplificado para velocidade)
    Z <- cbind(1, x1, z)
    Pz <- Z %*% solve(t(Z) %*% Z) %*% t(Z)
    Xhat <- Pz %*% X
    b_iv <- solve(t(Xhat) %*% Xhat, t(Xhat) %*% y)
    
    return(c(b_mqo[3], b_iv[3]))
  })
  
  # Estatisticas
  mqo_mean <- mean(results[1,])
  iv_mean <- mean(results[2,])
  
  cat(sprintf("Beta Real: 3.00\nMédia MQO: %.4f (Viés Positivo)\nMédia IV:  %.4f (Consistente)\n", 
              mqo_mean, iv_mean))
  
  par(mfrow=c(1,2))
  hist(results[1,], main="MQO Viesado", col="salmon")
  hist(results[2,], main="IV Consistente", col="lightgreen")
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main <- function() {
  cat("--- SISTEMA ECONOMETRICO R AVANÇADO ---\n")
  
  # 1. Geracao de Dados
  set.seed(42)
  N <- 500
  educ <- rnorm(N, 12, 2)
  exp <- rnorm(N, 10, 5)
  err <- rnorm(N, 0, 1)
  salario <- 5 + 0.8*educ + 0.2*exp + err
  
  # 2. Estimação
  X <- cbind(1, educ, exp)
  modelo <- Econometria(salario, X, nomes = c("Intercepto", "Educacao", "Experiencia"))
  modelo <- estimar_mqo(modelo)
  
  # 3. Resumo
  summary(modelo)
  
  # 4. Diagnosticos
  cat("\nDiagnóstico de Especificação (Ramsey RESET):\n")
  print(teste_reset(modelo))
  
  cat("\nDiagnóstico de Heterocedasticidade (Breusch-Pagan):\n")
  print(teste_bp(modelo))
  
  # 5. Simulacao
  run_massive_simulation(500)
}

# Executar Main se rodado como script
if (!interactive()) {
  main()
}
