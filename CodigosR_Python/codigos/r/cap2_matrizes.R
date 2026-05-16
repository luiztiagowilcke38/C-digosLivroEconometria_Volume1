# Operacoes Matriciais para Econometria

demonstrar_operacoes <- function() {
  # 1. Dados
  renda <- c(2000, 2500, 3000, 3500)
  consumo <- c(1500, 1800, 2100, 2400)
  
  # 2. Matriz X com intercepto
  n <- length(renda)
  X <- matrix(c(rep(1, n), renda), nrow = n, ncol = 2)
  
  # 3. XtX
  XtX <- t(X) %*% X
  
  # 4. Decomposicao QR
  qr_decomp <- qr(X)
  Q <- qr.Q(qr_decomp)
  R <- qr.R(qr_decomp)
  
  # 5. SVD
  svd_decomp <- svd(X)
  
  print("Matriz XtX:")
  print(XtX)
  print("Valores Singulares:")
  print(svd_decomp$d)
}

demonstrar_operacoes()
