library(plm)
# Usando dados simulados similares
df <- data.frame(id=rep(1:100, each=5), x=rnorm(500))
alpha <- rnorm(100)
df$x <- df$x + 0.5*alpha[df$id]
df$y <- 2.0*df$x + alpha[df$id] + rnorm(500)

# Efeitos Fixos
fe_mod <- plm(y ~ x, data=df, model="within", index=c("id"))

# Mundlak
df_means <- aggregate(x ~ id, data=df, mean)
names(df_means)[2] <- "x_mean"
df <- merge(df, df_means, by="id")
mun_mod <- lm(y ~ x + x_mean, data=df)

cat("FE beta: ", coef(fe_mod), "\n")
cat("Mundlak beta: ", coef(mun_mod)["x"], "\n")
