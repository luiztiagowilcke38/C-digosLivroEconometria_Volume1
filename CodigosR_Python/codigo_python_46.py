import dowhy
from dowhy import CausalModel
import pandas as pd
import numpy as np

# 1. Simular Dados Observacionais Estritamente Contaminados com Viés
np.random.seed(1)
# C(Confounder) -> X e C(Confounder) -> Y. Efeito Real de X em Y = 2.0
C = np.random.normal(50, 15, 1000)
X = 0.5 * C + np.random.normal(0, 5, 1000)
Y = 2.0 * X + 1.5 * C + np.random.normal(0, 10, 1000)
df = pd.DataFrame({'C': C, 'X': X, 'Y': Y})

# 2. Especificacao Textual do DAG Causal Assintotico
causal_graph = ''''''
digraph {
  C -> X;
  C -> Y;
  X -> Y;
}
''''''

# 3. Representação Universal Causal pelo Motor Estrutural da Biblioteca
model = CausalModel(
    data=df,
    treatment='X',
    outcome='Y',
    graph=causal_graph
)
model.view_model()

# 4. PASSO 1 - Identificacao Formal por Backdoor Criterion
identified_estimand = model.identify_effect()
print(identified_estimand)

# 5. PASSO 2 - Estimacao Frequencista e Validacao por Regressao Base
estimate = model.estimate_effect(identified_estimand,
                                 method_name=''backdoor.linear_regression'')
print(''Efeito Causal Identificado Analiticamente Pelo DoWhy:'')
print(estimate.value)

# 6. PASSO 3 - Refutacao Causal com Teste Placebo Randomico de Extrema Exigencia
refutation = model.refute_estimate(identified_estimand, estimate, 
                                   method_name=''random_common_cause'')
print(refutation)
