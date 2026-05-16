# Simulando 50 series temporais com 1 fator comum
n_series <- 50
n_obs <- 100
f_true <- rnorm(n_obs)
X <- matrix(NA, n_obs, n_series)
for(i in 1:n_series) X[,i] <- rnorm(1)*f_true + rnorm(n_obs, 0, 0.5)

# PCA para extrair o fator latente
pca_res <- prcomp(X, scale. = TRUE)
f_hat <- pca_res$x[,1] # Primeiro componente principal

plot(f_true, f_hat, main="Fator Real vs Estimado", xlab="Real", ylab="PCA")
