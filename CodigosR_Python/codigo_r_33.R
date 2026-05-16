library(boot)

# Funcao de estatistica para o boot
obter_beta <- function(data, indices) {
  d <- data[indices,]
  fit <- lm(y ~ x, data = d)
  return(coef(fit)[2]) # Retorna o slope
}

executar_bootstrap_rigoroso <- function(df) {
  resultados <- boot(data = df, statistic = obter_beta, R = 5000)
  
  # Intervalos de Confianca (Normal, Basico, Percentil, BCa)
  print(boot.ci(resultados, type = c(''norm'', ''perc'', ''bca'')))
  plot(resultados)
}
