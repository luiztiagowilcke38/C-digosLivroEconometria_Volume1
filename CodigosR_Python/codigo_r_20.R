library(gmm)
set.seed(1)
n <- 1000
z1 <- rnorm(n); z2 <- rnorm(n)
u <- rnorm(n)
x <- 0.5*z1 + 0.5*z2 + 0.5*u + rnorm(n, 0, 0.5)
y <- 2.0*x + u

# Definindo momentos: g = Z'(y - Xb)
# No R, passamos a formula e os instrumentos separadamente
res_gmm <- gmm(y ~ x, x = ~ z1 + z2, type = "twoStep")
summary(res_gmm)
