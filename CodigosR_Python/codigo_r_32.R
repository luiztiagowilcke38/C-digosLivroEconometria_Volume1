library(mlogit)

analise_multinomial <- function(df) {
  # Preparando dados para formato long (exigido pelo mlogit)
  data_long <- mlogit.data(df, choice = ''brand'', shape = ''wide'')
  
  # Estimacao Proportional Odds ou Multinomial Logit
  fit <- mlogit(brand ~ price + feat | 0, data = data_long)
  
  # Teste de Independencia de Alternativas Irrelevantes (IIA)
  # Teste de Hausman-McFadden
  print(summary(fit))
}
