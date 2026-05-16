from spreg import OLS
from spreg.diagnostics_sp import LMtests

# Regressao MQO simples como ponto de partida
ols = OLS(y.reshape(-1,1), X_mat, w,
          name_y='Y', name_x=['X1','X2'],
          spat_diag=True)
print(ols.summary)

# Os testes LM sao automaticamente incluídos com spat_diag=True
# Interprete: LM-Lag >> LM-Error => SAR eh preferivel
# Interprete: RLM-Lag < RLM-Error => ainda favoravel para SAR
