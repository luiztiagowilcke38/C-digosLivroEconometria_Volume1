library(splm)
library(spdep)

# Carregar dados de painel: 49 estados US, 17 anos
data(Produc, package=''plm'')

# Criar listw de vizinhança
data(nb_states, package=''splm'')
lw <- nb2listw(nb_states, style=''W'')

# Modelo de Efeitos Fixos Espaciais (SAR within)
spfe <- spml(log(gsp) ~ log(pcap) + log(pc) + log(emp) + unemp,
             data = Produc, listw = lw,
             model = ''within'',
             spatial.error = ''none'',
             lag = TRUE)
summary(spfe)

# Impactos Parciais no Painel
im <- impacts(spfe, listw = lw, time = 17)
summary(im, zstats=TRUE)
