from sklearn.model_selection import KFold
from sklearn.base import clone

def dml_cross_fitting(y, D, X, model_y, model_d, K=5):
    kf = KFold(n_splits=K, shuffle=True)
    res_y = np.zeros_like(y)
    res_d = np.zeros_like(D)
    
    for train_idx, test_idx in kf.split(X):
        # 1. Clonar modelos para evitar vazamento de dados
        m_y, m_d = clone(model_y), clone(model_d)
        
        # 2. Treinar no Fold de Treino
        m_y.fit(X[train_idx], y[train_idx])
        m_d.fit(X[train_idx], D[train_idx])
        
        # 3. Prever no Fold de Teste (Out-of-sample)
        res_y[test_idx] = y[test_idx] - m_y.predict(X[test_idx])
        res_d[test_idx] = D[test_idx] - m_d.predict(X[test_idx])
        
    # Estimação final por MQO
    tau = np.dot(res_d, res_y) / np.dot(res_d, res_d)
    return tau
