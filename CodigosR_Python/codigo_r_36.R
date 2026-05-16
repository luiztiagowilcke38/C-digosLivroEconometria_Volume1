mqo_estavel <- function(y, X) {
  # Decomposicao QR: X = QR
  res_qr <- qr(X)
  
  # b = R^-1 Q' y
  beta <- solve.qr(res_qr, y)
  
  # Residuos
  residuos <- y - X %*% beta
  sigma2 <- sum(residuos^2) / (nrow(X) - ncol(X))
  
  # Erros Padrao via R (parte da decomposicao QR)
  R_mat <- qr.R(res_qr)
  XpX_inv <- solve(t(R_mat) %*% R_mat)
  se <- sqrt(diag(sigma2 * XpX_inv))
  
  return(list(beta=beta, se=se))
}
