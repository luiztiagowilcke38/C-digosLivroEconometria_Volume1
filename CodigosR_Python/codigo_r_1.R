library(MatchIt)
library(lmtest)
library(sandwich)

data(''lalonde'', package=''MatchIt'')

# Estimar PS Score via Logit
m.out <- matchit(treat ~ age + educ + race + married + nodegree + re74 + re75,
                 data = lalonde, method = ''nearest'', distance = ''logit'',
                 replace = FALSE, caliper = 0.05)

# Diagnosticar Suporte Comum e Balanceamento
plot(m.out, type = ''jitter'')
plot(summary(m.out))

# Extrair Dados Pareados
m.data <- match.data(m.out)

# Estima ATT Robusto usando WLS 
fit <- lm(re78 ~ treat, data = m.data, weights = weights)

# Computa HC3 Robusto para Inferencia Valida
coeftest(fit, vcov. = vcovHC(fit, type=''HC3''))
