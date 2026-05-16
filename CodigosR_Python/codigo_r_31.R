library(lmtest)
library(sandwich)

analise_robusta <- function(modelo) {
  # 1. Teste de Breusch-Pagan
  print(bptest(modelo))
  
  # 2. Erros Padrao de White (HC3)
  print(coeftest(modelo, vcov = vcovHC(modelo, type = ''HC3'')))
  
  # 3. Newey-West para Series Temporais
  print(coeftest(modelo, vcov = NeweyWest(modelo, lag = 4)))
}
