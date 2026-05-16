library(AER)

analise_iv_completa <- function(formula_iv, df) {
  # Exemplo de formula: y ~ x_endo + x_exo | z_inst + x_exo
  modelo_iv <- ivreg(formula_iv, data = df)
  
  # Diagnosticos Automaticos
  # Inclui: Weak Instruments, Wu-Hausman e Sargan
  print(summary(modelo_iv, diagnostics = TRUE))
}
