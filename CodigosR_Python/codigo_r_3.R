library(DoubleML)
library(mlr3)
library(mlr3learners)
library(data.table)

estimar_efeito_causal_dml <- function(dados_DT, treatment_col, outcome_col, covariates_cols) {
  # Instanciacao do Objeto DML Data Engine 
  dml_data <- DoubleMLData$new(dados_DT, 
                               y_col = outcome_col, 
                               d_cols = treatment_col, 
                               x_cols = covariates_cols)
  
  # Especificacao de Learners nao parametricos de Fronteira (Random Forests - Ranger)
  # Usaremos 500 arvores para estimacao das esperancas condicionais
  learner_g <- lrn(''regr.ranger'', num.trees = 500, max.depth=10, predict_type = ''response'')
  learner_m <- lrn(''regr.ranger'', num.trees = 500, max.depth=10, predict_type = ''response'')
  
  # DML Modelo Parcial Linear Re-amostrado Cross-fitted (PLR)
  # ATE (Average Treatment Effect) Equation: Y = D*theta + g(X) + U
  dml_plr <- DoubleMLPLR$new(dml_data, 
                             ml_g = learner_g, 
                             ml_m = learner_m, 
                             n_folds = 5, # 5-Fold Cross Fitting
                             score = ''partialling out'') # Escores de Neyman Ortogonal
  
  # Otimizacao Multicore Integrada Future Backend
  library(future)
  plan(multisession, workers = 4)
  
  # Ajuste com Otimizacao iterativa OLS pos-residual
  dml_plr$fit()
  
  print(dml_plr$summary())
  
  # Inferencia Gaussiana Pos-Processada
  confint_matrix <- dml_plr$confint(level = 0.95)
  return(list(Estimativa = dml_plr$coef, IC=confint_matrix))
}

# resultado_causal <- estimar_efeito_causal_dml(dt, ''Tratamento'', ''Renda'', n_covariadas_100_)
