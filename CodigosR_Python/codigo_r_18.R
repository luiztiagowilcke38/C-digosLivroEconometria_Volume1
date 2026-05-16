set.seed(42)
n <- 300
x1 <- rnorm(n); x2 <- rnorm(n)
y  <- 2.0 + 1.5*x1 - 0.8*x2 + rnorm(n, 0, 1.5)

# Funcao auxiliar: decomposicao manual SST/SSR/SSE
ols_r2 <- function(model){
    y_obs <- model$model[,1]
    y_hat <- fitted(model)
    resid <- residuals(model)
    SST <- sum((y_obs - mean(y_obs))^2)
    SSE <- sum(resid^2)
    SSR <- SST - SSE
    n_  <- length(y_obs)
    k_  <- length(coef(model)) - 1
    R2  <- 1 - SSE/SST
    R2adj <- 1 - (SSE/(n_-k_-1)) / (SST/(n_-1))
    cat(sprintf(''R2=%.4f  R2adj=%.4f  SST=%.1f  SSR=%.1f  SSE=%.1f\n'',
                R2, R2adj, SST, SSR, SSE))
    invisible(list(R2=R2, R2adj=R2adj))
}

# =========================================================
# PARTE A: R2 com diferentes numeros de regressores
# =========================================================
cat(''--- Modelos com 0/1/2 regressores relevantes ---\n'')
df <- data.frame(y=y, x1=x1, x2=x2)
cat(''Nenhum regressor: ''); ols_r2(lm(y ~ 1,     data=df))
cat(''So x1:            ''); ols_r2(lm(y ~ x1,    data=df))
cat(''x1 e x2:          ''); ols_r2(lm(y ~ x1+x2, data=df))

# =========================================================
# PARTE B: R2 nunca diminui com mais variaveis
# =========================================================
cat(''\n--- R2 ao adicionar variaveis aleatorias ---\n'')
df_ext <- df
for(j in 1:10){
    df_ext[[paste0(''ruido'',j)]] <- rnorm(n)
    vars  <- paste(c(''x1'',''x2'',paste0(''ruido'',1:j)), collapse=''+'')
    form  <- as.formula(paste(''y ~'', vars))
    mod_j <- lm(form, data=df_ext)
    sm    <- summary(mod_j)
    cat(sprintf(''  k=%2d: R2=%.5f  R2adj=%.5f\n'',
                length(coef(mod_j))-1, sm$r.squared, sm$adj.r.squared))
}

# =========================================================
# PARTE C: Regressao Espuria - dois passeios aleatorios
# =========================================================
cat(''\n--- Regressao Espuria ---\n'')
T    <- 500
w1   <- cumsum(rnorm(T))
w2   <- cumsum(rnorm(T))
mod_esp <- lm(w2 ~ w1)
sm_esp  <- summary(mod_esp)

# Durbin-Watson manual
e    <- residuals(mod_esp)
dw   <- sum(diff(e)^2) / sum(e^2)
cat(sprintf(''R2 espurio: %.4f | DW: %.4f | p(t-stat): %.4f\n'',
            sm_esp$r.squared, dw, sm_esp$coefficients[2,4]))
cat(''Interpretacao: R2 alto + DW << 2 = regressao espuria classica\n'')

# =========================================================
# PARTE D: R2 sem intercepto
# =========================================================
cat(''\n--- R2 sem intercepto (interpretacao incorreta) ---\n'')
mod_no_int <- lm(y ~ x1 - 1)   # -1 remove intercepto
SSE_ni  <- sum(residuals(mod_no_int)^2)
SST_med <- sum((y - mean(y))^2)   # base convencional
SST_0   <- sum(y^2)               # base correta (referencia = 0)
cat(sprintf(''R2 (base media, inadequado): %.4f\n'', 1 - SSE_ni/SST_med))
cat(sprintf(''R2 (base zero, adequado):    %.4f\n'', 1 - SSE_ni/SST_0))
cat(''summary() do R usa base zero automaticamente quando remove intercepto.\n'')

# =========================================================
# PARTE E: Criterios ALTERNATIVOS ao R2 para selecao de modelos
# =========================================================
cat(''\n--- Criterios AIC e BIC (alternativas ao R2adj) ---\n'')
mods <- list(
    M1 = lm(y ~ x1,                 data=df_ext),
    M2 = lm(y ~ x1 + x2,           data=df_ext),
    M3 = lm(y ~ x1 + x2 + ruido1,  data=df_ext),
    M4 = lm(y ~ x1 + x2 + ruido1 + ruido2 + ruido3, data=df_ext)
)
for(nm in names(mods)){
    sm <- summary(mods[[nm]])
    cat(sprintf(''%s: R2=%.4f R2adj=%.4f AIC=%.2f BIC=%.2f\n'',
                nm, sm$r.squared, sm$adj.r.squared,
                AIC(mods[[nm]]), BIC(mods[[nm]])))
}
cat(''BIC penaliza mais fortemente variaveis desnecessarias que AIC.\n'')
