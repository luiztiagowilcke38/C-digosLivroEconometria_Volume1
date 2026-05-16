set.seed(99)
n <- 3000
C <- rnorm(n)
X <- 0.6*C + rnorm(n, 0, sqrt(1-0.36))
Y <- 2.0*X + 1.5*C + rnorm(n)
df <- data.frame(Y=Y, X=X, C=C)

# (a) MQO ingênuo
cat(''MQO sem controle:'', coef(lm(Y~X, data=df))[2], ''\n'')

# (b) Ajuste Back-Door controlando por C
mod <- lm(Y ~ X + C, data=df)
cat(''ATE ajustado:    '', coef(mod)[2], ''\n'')
print(confint(mod)[''X'',])

# (c) G-Computation via Marginalizacao Numerica Integral
# Se X nao-linear: prever E[Y|do(X=x)] = mean_over_C(E[Y|X=x, C])
x_grid <- seq(min(X), max(X), length.out=50)
ate_gc <- mean(sapply(x_grid, function(xv){
    # Prever Y para todos os individuos mudando X para xv, mantendo C real
    mean(predict(mod, newdata=data.frame(X=xv, C=C)))
}))
# Derivative numerica: d/dx E[Y|do(X=x)]
dx <- x_grid[2] - x_grid[1]
y_pred_grid <- sapply(x_grid, function(xv) mean(predict(mod, newdata=data.frame(X=xv, C=C))))
ace_numeric <- mean(diff(y_pred_grid) / dx)
cat(sprintf(''ACE numerico (G-Computation): %.3f\n'', ace_numeric))
