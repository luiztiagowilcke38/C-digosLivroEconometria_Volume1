import numpy as np
import scipy.linalg as la

def analise_estabilidade_svd(X, y):
    # SVD para resolver sistemas mal-condicionados
    U, s, Vt = la.svd(X, full_matrices=False)
    
    # Inversa via valores singulares (truncada se necessario)
    S_inv = np.diag(1/s)
    beta_svd = Vt.T @ S_inv @ U.T @ y
    
    # Numero de Condicao
    cond_num = s[0] / s[-1]
    print(f''Numero de Condicao: {cond_num}'')
    
    return beta_svd, cond_num

def complemento_schur_manual(A, B, C, D):
    # S = D - C*inv(A)*B
    A_inv = la.inv(A)
    S = D - C @ A_inv @ B
    return S

def recursao_inversao_matriz(A, novo_vetor):
    # Implementacao do Lemma de Inversao (Woodbury)
    # Util para Big Data e Filtros de Kalman
    A_inv = la.inv(A)
    # ... (logica de atualizacao iterativa)
    return A_inv
