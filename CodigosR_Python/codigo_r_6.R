library(vars)

executar_var_completo <- function(data_matrix) {
  # 1. Selecao de Lag Otimo (AIC, BIC)
  lag_selection <- VARselect(data_matrix, lag.max = 10, type = ''both'')
  p_opt <- lag_selection$selection[''AIC(n)'']
  
  # 2. Estimacao
  var_est <- VAR(data_matrix, p = p_opt, type = ''both'')
  
  # 3. Funcao de Impulso-Resposta
  irf_res <- irf(var_est, impulse = ''PIB'', response = ''Inflacao'', n.ahead = 10)
  plot(irf_res)
  
  # 4. Causalidade de Granger
  print(causality(var_est, cause = ''Taxa_Juros''))
}
