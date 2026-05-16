library(mlogit)

# No R, mlogit exige um formato de dados longo (dfidx)
# Simulacao simples
set.seed(42)
n <- 500
x <- rnorm(n)
# Probabilidades
v2 <- 0.5 + 1.2*x
v3 <- -0.8 + 2.5*x
ev1 <- 1; ev2 <- exp(v2); ev3 <- exp(v3); sume <- ev1+ev2+ev3
p1 <- ev1/sume; p2 <- ev2/sume; p3 <- ev3/sume
y <- apply(cbind(p1,p2,p3), 1, function(p) sample(1:3, 1, prob=p))

df <- data.frame(choice=y, x=x, id=1:n)
df_idx <- dfidx(df, choice="choice", shape="wide", varying=NULL)

# Estimacao
fit <- mlogit(choice ~ 1 | x, data=df_idx, reflevel="1")
summary(fit)
