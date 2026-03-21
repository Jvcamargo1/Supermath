import numpy as np
from scipy.linalg import lu as scipy_lu
from scipy.linalg import solve_triangular

# Funções de Fatoração LU
def fatoracao_lu(A):
    """
    Realiza a Fatoração PLU de uma matriz A (PA = LU).
    
    Args:
        A: A matriz a ser decomposta (numpy array).
        
    Returns:
        As matrizes P (permutação), L (triangular inferior) e U (triangular superior).
        Retorna None, None, None se a fatoração falhar.
    """
    try:
        P, L, U = scipy_lu(A)
        return P, L, U
    except (np.linalg.LinAlgError, ValueError):
        return None, None, None

def solve_lu(P, L, U, b):
    """
    Resolve um sistema linear Ax=b dada a Fatoração PLU de A.
    
    Args:
        P: Matriz de permutação.
        L: Matriz triangular inferior.
        U: Matriz triangular superior.
        b: Vetor de termos independentes.
        
    Returns:
        O vetor solução x.
    """
    # Ax = b  =>  PAx = Pb  =>  LUx = Pb
    # 1. Resolva Ly = Pb  (forward substitution)
    Pb = np.dot(P, b)
    y = solve_triangular(L, Pb, lower=True, unit_diagonal=False)
    
    # 2. Resolva Ux = y  (backward substitution)
    x = solve_triangular(U, y, lower=False, unit_diagonal=False)
    
    return x
