library(ggplot2)
library(broom)

visualizar_inferencia <- function(modelo) {
  # Extrair coeficientes e intervalos de confianca
  td <- tidy(modelo, conf.int = TRUE)
  
  ggplot(td, aes(x = estimate, y = term)) +
    geom_point(size = 3) +
    geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2) +
    geom_vline(xintercept = 0, color = ''red'', linetype = ''dashed'') +
    labs(title = ''Coefplot: Inferencia e Causalidade'',
         x = ''Estimativa do Parametro'', y = ''Regressores'') +
    theme_minimal()
}
