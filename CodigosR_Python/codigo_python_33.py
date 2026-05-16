import numpy as np

def estimar_2sls_manual(y, X, Z):
    # n: obs, k: regressores em X, l: instrumentos em Z
    # Z deve incluir os regressores exotenos de X
    
    # 1. Primeiro Estagio: X_hat = Pz * X
    # Pz = Z * (Z'Z)^-1 * Z'
    Pz = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    X_hat = Pz @ X
    
    # 2. Segundo Estagio: beta = (X_hat' * X_hat)^-1 * X_hat' * y
    beta_iv = np.linalg.inv(X_hat.T @ X_hat) @ X_hat.T @ y
    
    # 3. Residuos (usando X real, nao X_hat!)
    residuos = y - X @ beta_iv
    sigma2 = (residuos.T @ residuos) / (len(y) - X.shape[1])
    
    # 4. Variancia de Beta_IV
    var_iv = sigma2 * np.linalg.inv(X_hat.T @ X_hat)
    return beta_iv, np.sqrt(np.diag(var_iv))

# Z_total = np.column_stack([z_inst, x_exog])
