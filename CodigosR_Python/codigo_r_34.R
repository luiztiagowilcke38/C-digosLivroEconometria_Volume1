library(Synth)

# Simulação de Painel
n_units <- 10
n_periods <- 20
treated_unit <- 1
treatment_period <- 11

set.seed(42)
Y <- matrix(rnorm(n_units * n_periods), n_units, n_periods)
# Efeito do tratamento na unidade 1 a partir do periodo 11
Y[treated_unit, treatment_period:n_periods] <- Y[treated_unit, treatment_period:n_periods] + 2.0

# Organizando para o pacote Synth
data <- data.frame(
  id = rep(1:n_units, each=n_periods),
  year = rep(1:n_periods, n_units),
  outcome = as.vector(t(Y))
)

# Preparando os dados
dataprep_out <- dataprep(
  foo = data,
  predictors = c("outcome"),
  predictors.op = "mean",
  dependent = "outcome",
  unit.variable = "id",
  time.variable = "year",
  treatment.identifier = 1,
  controls.identifier = 2:n_units,
  time.predictors.prior = 1:10,
  time.optimize.ssr = 1:10,
  unit.names.variable = "id",
  time.plot = 1:20
)

# Ouro: Calculando os pesos sinteticos
synth_out <- synth(dataprep_out)

# Visualizando o Gap Causal
path.plot(synth_out, dataprep_out)
