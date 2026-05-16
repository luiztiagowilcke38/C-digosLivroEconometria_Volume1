import geopandas as gpd
import numpy as np
import libpysal.weights as weights
from spreg import ML_Lag, ML_Error
from shapely.geometry import Polygon

# 1. Gerar Grid Espacial Sintetica (10x10 regioes)
cols, rows = 10, 10
polygons = []
for i in range(cols):
    for j in range(rows):
        polygons.append(Polygon([(i,j), (i+1,j), (i+1,j+1), (i,j+1)]))

gdf = gpd.GeoDataFrame(geometry=polygons)

# Gerar dados
np.random.seed(42)
gdf['X1'] = np.random.normal(10, 2, 100)
gdf['X2'] = np.random.uniform(0, 5, 100)
e = np.random.normal(0, 1, 100)

# 2. Construcao da Matriz Queen normalizada
w = weights.Queen.from_dataframe(gdf)
w.transform = 'r' # Row-standardization

# 3. Gerar Processo SAR
rho = 0.6
I = np.eye(100)
W_matrix = w.full()[0]
beta = np.array([5.0, 2.5]) # Efeitos verdadeiros
X_mat = np.column_stack((gdf['X1'], gdf['X2']))

inv_mat = np.linalg.inv(I - rho * W_matrix)
y = inv_mat @ (X_mat @ beta + e)
gdf['Y'] = y

# 4. Estimacao Master ML-SAR
sar_model = ML_Lag(y.reshape(-1,1), X_mat, w, name_y='Y', name_x=['X1', 'X2'])
print(sar_model.summary)

# Impactos via S_r(W) - Abordagem Analitica Reta
rho_hat = sar_model.rho
beta_hat = sar_model.betas[1:3] # ignora const
inv_mat_hat = np.linalg.inv(I - rho_hat * W_matrix)

S_x1 = inv_mat_hat * beta_hat[0]
direct = np.trace(S_x1)/100
total = np.sum(S_x1)/100
indirect = total - direct
print(f''Impacto Direto X1: {direct:.4f}'')
print(f''Impacto Indireto X1: {indirect:.4f}'')
