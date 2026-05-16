#' Suíte de Econometria Avançada - Luiz Tiago Wilcke
#' Autor: Luiz Tiago Wilcke
#' Descrição: Implementação de modelos econométricos de fronteira em R:
#'              GMM, Econometria Espacial, Painel Dinâmico e Descontinuidade.
#'              Utiliza classes S3 para uma estrutura profissional.

library(stats)
library(gmm)
library(spatialreg)
library(spdep)
library(plm)
library(MatchIt)

# ==============================================================================
# 1. MÉTODO DOS MOMENTOS GENERALIZADOS (GMM)
# ==============================================================================

#' Construtor para Modelo GMM
#' @param y Vetor dependente
#' @param x Matriz de regressores
#' @param z Matriz de instrumentos
ModeloGMM <- function(y, x, z) {
  modelo <- list(y = y, x = x, z = z, resultados = NULL)
  class(modelo) <- "ModeloGMM"
  return(modelo)
}

#' Ajustar modelo GMM
ajustar <- function(obj, ...) UseMethod("ajustar")

ajustar.ModeloGMM <- function(obj) {
  # Função de momentos: E[z' * (y - x*beta)] = 0
  g <- function(beta, dat) {
    y <- dat[, 1]
    x <- dat[, 2:(ncol(obj$x)+1)]
    z <- dat[, (ncol(obj$x)+2):ncol(dat)]
    residuos <- y - x %*% beta
    momento <- z * as.numeric(residuos)
    return(momento)
  }
  
  dat_gmm <- cbind(obj$y, obj$x, obj$z)
  res <- gmm(g, dat_gmm, t0 = rep(0, ncol(obj$x)), wmatrix = "optimal", optfct = "optim")
  
  obj$resultados <- res
  return(obj)
}

# ==============================================================================
# 2. ECONOMETRIA ESPACIAL (SAR e Impactos)
# ==============================================================================

#' Construtor para Econometria Espacial
ModeloEspacial <- function(y, x, listw) {
  modelo <- list(y = y, x = x, listw = listw, fit = NULL)
  class(modelo) <- "ModeloEspacial"
  return(modelo)
}

estimar_sar <- function(obj) {
  # Estima o modelo Spatial Autoregressive (SAR)
  # y = rho*W*y + X*beta + e
  formula_sar <- y ~ .
  dados <- data.frame(y = obj$y, obj$x)
  
  fit <- lagsarlm(formula_sar, data = dados, listw = obj$listw)
  obj$fit <- fit
  return(obj)
}

calcular_impactos <- function(obj) {
  if (is.null(obj$fit)) stop("Modelo deve ser estimado primeiro.")
  # Impactos Diretos, Indiretos e Totais (LeSage & Pace)
  imp <- impacts(obj$fit, listw = obj$listw, R = 500)
  return(imp)
}

# ==============================================================================
# 3. PAINEL DINÂMICO (Arellano-Bond)
# ==============================================================================

#' Estimar Arellano-Bond usando plm
estimar_arellano_bond <- function(dados, formula_eb, id_col, tempo_col) {
  # y ~ lag(y, 1) + x1 + x2
  # GMM de diferença (Arellano-Bond 1991)
  p_dados <- pdata.frame(dados, index = c(id_col, tempo_col))
  
  fit <- pgmm(formula_eb, 
              data = p_dados, 
              effect = "individual", 
              model = "onestep", 
              transformation = "ld")
  return(fit)
}

# ==============================================================================
# 4. MICROECONOMETRIA CAUSAL (RDD e PSM)
# ==============================================================================

#' Propensity Score Matching
executar_psm <- function(dados, trat_col, cov_cols) {
  form <- as.formula(paste(trat_col, "~", paste(cov_cols, collapse = "+")))
  m_out <- matchit(form, data = dados, method = "nearest", distance = "logit")
  return(m_out)
}

# ==============================================================================
# DEMONSTRAÇÃO DOS MODELOS
# ==============================================================================

executar_demonstracao <- function() {
  cat("\n==========================================================\n")
  cat("DEMONSTRAÇÃO DE ECONOMETRIA CORE - R\n")
  cat("Autor: Luiz Tiago Wilcke\n")
  cat("==========================================================\n")
  
  # 1. GMM
  n <- 500
  z <- matrix(rnorm(n*2), ncol=2)
  x <- 0.7 * z[,1] + rnorm(n, 0, 0.5)
  y <- 3.0 * x + rnorm(n)
  
  mod_gmm <- ModeloGMM(y, as.matrix(x), z)
  ajustado <- ajustar(mod_gmm)
  cat("\n[GMM] Estimativa de Beta:", coef(ajustado$resultados), "\n")
  
  # 2. Espacial
  # Criando matriz de pesos simples (cadeia)
  nb <- cell2nb(5, 5) # Grade 5x5
  lw <- nb2listw(nb, style="W")
  
  x_esp <- rnorm(25)
  rho <- 0.6
  y_esp <- invIrW(lw, rho) %*% (x_esp * 2.5 + rnorm(25))
  
  mod_esp <- ModeloEspacial(y_esp, data.frame(X1 = x_esp), lw)
  estimado <- estimar_sar(mod_esp)
  cat("\n[Espacial SAR] Rho Estimado:", estimado$fit$rho, "\n")
  
  impactos <- calcular_impactos(estimado)
  cat("[Espacial SAR] Impacto Total X1:", impactos$total, "\n")
  
  # 3. Arellano-Bond (Simulação rápida)
  set.seed(42)
  n_id <- 20; n_t <- 10
  id <- rep(1:n_id, each=n_t)
  tempo <- rep(1:n_t, n_id)
  y_pan <- rnorm(n_id*n_t)
  x_pan <- rnorm(n_id*n_t)
  df_pan <- data.frame(id=id, tempo=tempo, y=y_pan, x=x_pan)
  
  # Nota: pgmm requer estrutura de dados específica e lags
  # fit_ab <- estimar_arellano_bond(df_pan, y ~ lag(y, 1) + x | lag(y, 2:99), "id", "tempo")
  # cat("\n[Arellano-Bond] Modelo instanciado com sucesso.\n")
}

if (!interactive()) {
  executar_demonstracao()
}

# --- COMENTÁRIOS DE RIGOR ECONOMÉTRICO ---
#' DISCUSSÃO TEÓRICA:
#' 
#' O estimador de Arellano-Bond (GMM-Diff) é fundamental para tratar a 
#' endogeneidade causada pela presença de efeitos fixos correlacionados com a 
#' defasagem da variável dependente. No entanto, em amostras com persistência 
#' muito alta, o GMM-System (Blundell e Bond) é preferível para evitar o 
#' problema de instrumentos fracos.
#' 
#' Na Econometria Espacial, o modelo SARAR (ou SAC) é o mais completo ao 
#' permitir dependência tanto na variável observada quanto no termo de erro, 
#' capturando transbordamentos complexos e correlações espaciais omitidas.
