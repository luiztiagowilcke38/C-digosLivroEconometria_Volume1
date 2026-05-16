library(grf)

estimar_efeitos_heterogeneos <- function(Y, D, X) {
  # Treina uma Causal Forest
  c_forest <- causal_forest(X, Y, D)
  
  # Estima o Conditional Average Treatment Effect (CATE)
  tau_hat <- predict(c_forest, X)
  
  # Teste de heterogeneidade de Larkey
  test_calibration(c_forest)
  
  hist(tau_hat$predictions, main=''Distribuicao do Efeito Causal'', 
       xlab=''Tau estimado'', col=''lightblue'')
}
