import shap
from sklearn.ensemble import RandomForestRegressor

# Treinamento de um Modelo Genérico (Ex: Previsão de Renda)
model = RandomForestRegressor(n_estimators=100, max_depth=5).fit(X_train, y_train)

# Cálculo dos Valores SHAP (TreeExplainer)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualização de Resumo (Summary Plot)
shap.summary_plot(shap_values, X_test)

# Impacto local para uma observação i
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test[0,:])
