library(quantreg)
set.seed(7)
n <- 200
x <- rnorm(n)
y <- 1.0 + 2.5*x + 0.5*rnorm(n)*(1 + abs(rnorm(n)))
df <- data.frame(y=y, x=x)

# Varios quantis simultaneamente
taus <- c(0.10, 0.25, 0.50, 0.75, 0.90)
fit_rq <- rq(y ~ x, tau=taus, data=df)
print(coef(fit_rq))

# Teste de simetria (Ha: efeitos iguais nos dois lados)
anova_test <- anova(rq(y ~ x, tau=0.25, data=df),
                    rq(y ~ x, tau=0.75, data=df))
print(anova_test)

# Intervalo de Confianca por Inversao de Rank Test (Koenker-Bassett)
summary_q75 <- summary(rq(y ~ x, tau=0.75, data=df), se=''rank'')
print(summary_q75)
