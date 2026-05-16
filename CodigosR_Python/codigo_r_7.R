library(tidyverse)
library(stargazer)

gerar_relatorio_tecnico <- function(raw_data) {
  # Pipe de Limpeza
  df_clean <- raw_data %>%
    mutate(exper2 = exper^2) %>%
    filter(income > 0)
    
  # Modelos
  fit_ols <- lm(log(income) ~ exper + exper2 + gender, data = df_clean)
  
  # Saida LaTeX para o Manuscrito
  stargazer(fit_ols, type = ''latex'', 
            title = ''Resultados da Regressao de Mincer'',
            header = FALSE, font.size = ''small'')
}
