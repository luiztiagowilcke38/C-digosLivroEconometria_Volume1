library(DoubleML)
library(mlr3)
library(mlr3learners)
library(data.table)

# Simulacao similar
set.seed(42)
n <- 1000
k <- 20
X <- matrix(rnorm(n*k), n, k)
colnames(X) <- paste0("X", 1:k)
D <- sin(X[,1]) + 0.5*X[,2]^2 + rnorm(n)
Y <- 2.5*D + exp(X[,3]) + X[,4]*X[,5] + rnorm(n)
dt <- as.data.table(cbind(Y, D, X))

# Definindo o objeto de dados
dml_data <- double_ml_data_from_data_frame(dt, y_col = "V1", d_cols = "V2")

# Selecionando learners (Random Forest do mlr3)
l_y <- lrn("regr.ranger")
l_d <- lrn("regr.ranger")

# Estimando DML (PLR - Partially Linear Regression)
dml_plr <- DoubleMLPLR$new(dml_data, l_y, l_d, n_folds = 2)
dml_plr$fit()
print(dml_plr)
