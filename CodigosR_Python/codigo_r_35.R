library(quantreg)

analisar_distribuicao <- function(df) {
  # Estima varios quantis simultaneamente (0.1, 0.25, 0.5, 0.75, 0.9)
  fit_multi <- rq(wage ~ educ + exper, tau = c(0.1, 0.25, 0.5, 0.75, 0.9), data = df)
  
  # Plot dos coeficientes mudando ao longo dos quantis
  plot(summary(fit_multi))
  
  # Teste de Anova entre os quantis (Os efeitos sao iguais?)
  anova(rq(wage ~ educ, tau=0.25, data=df), rq(wage ~ educ, tau=0.75, data=df))
}
