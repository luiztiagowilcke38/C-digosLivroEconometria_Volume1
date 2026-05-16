library(copula)
library(lattice)

simular_dependencia_extrema <- function() {
  # Copula de Clayton captura dependencia nas caudas inferiores
  myCop <- claytonCopula(param = 2, dim = 2)
  myMvd <- mvdc(copula = myCop, margins = c(''norm'', ''exp''), 
                paramMargins = list(list(mean = 0, sd = 1), list(rate = 2)))
  
  dados <- rMvdc(2000, myMvd)
  plot(dados, main=''Dependencia de Cauda via Copula de Clayton'', 
       xlab=''Normal'', ylab=''Exponencial'', col=rgb(0,0,1,0.3), pch=19)
}
