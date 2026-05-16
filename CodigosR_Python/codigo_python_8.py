import numpy as np
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
import libpysal

# Dados de Criminalidade de Columbus Ohio
data = libpysal.examples.load_example('Columbus')
df_col = libpysal.io.open(data.get_path('columbus.dbf')).read()

y_gwr = np.array(df_col['CRIME']).reshape(-1,1)
X_gwr = np.array([df_col['INC'], df_col['HOVAL']]).T

# Coordenadas centróides
from libpysal.examples import get_path
coords = np.array(list(zip(df_col['X'], df_col['Y'])))

# Selecao Automatica de Bandwidth via Golden Section Search
selector = Sel_BW(coords, y_gwr, X_gwr)
bw_opt = selector.search()
print(f'Bandwidth otimo: {bw_opt}')

# Estimando GWR com Kernel Gaussiano Adaptivo
gwr_model = GWR(coords, y_gwr, X_gwr, bw=bw_opt, kernel='gaussian', fixed=False)
gwr_results = gwr_model.fit()
print(gwr_results.summary())
