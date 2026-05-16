library(spdep)
library(spatialreg)
library(sf)

# Criando grade sf
grid <- st_make_grid(st_sfc(st_point(c(0,0)), st_point(c(10,10))), n=c(10,10))
data_esp <- st_sf(ID = 1:100, geometry = grid)
data_esp$X1 <- rnorm(100, 5, 1)

# Lista de vizinhos tipo Rook e array Lw
nb <- poly2nb(data_esp, queen=FALSE)
lw <- nb2listw(nb, style=''W'')

# Simulacao Durbin em R (SAC mix)
data_esp$WX1 <- lag.listw(lw, data_esp$X1)
error <- rnorm(100)
spat_err <- invIrW(lw, 0.5) %*% error # lambda = 0.5

# Y = rho W Y + X B + WX T + erro espacial(lambda)
# Simulando apenas SDM
data_esp$Y <- invIrW(lw, 0.3) %*% (2*data_esp$X1 + 1.5*data_esp$WX1 + error)

# Estimando SDM (Spatial Durbin Model)
mod_sdm <- lagsarlm(Y ~ X1, data=data_esp, listw=lw, type=''mixed'')
summary(mod_sdm)

# Computando os Impactos com Simulacao Monte Carlo para Standard Errors
impl <- impacts(mod_sdm, listw=lw, R=1000)
print(impl)
summary(impl, zstats=TRUE, short=TRUE)
