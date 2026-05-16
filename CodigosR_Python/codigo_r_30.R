simular_correlacionados <- function(n, rho) {
  # Matriz de correlacao 2x2
  Sigma <- matrix(c(1, rho, rho, 1), 2, 2)
  
  # Fator de Cholesky
  L <- chol(Sigma) # R retorna triangular superior
  
  # Gerando dados independentes
  Z <- matrix(rnorm(n*2), n, 2)
  
  # Transformando
  X <- Z %*% L
  return(X)
}
