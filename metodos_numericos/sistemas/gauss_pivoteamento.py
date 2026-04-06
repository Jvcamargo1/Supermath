import numpy as np

# Funções de Eliminação de Gauss com Pivoteamento
def gauss_pivoteamento(A, b):
    """
    Resolve um sistema linear Ax=b usando Eliminação de Gauss com Pivoteamento Parcial.
    
    Args:
        A: Matriz de coeficientes (numpy array).
        b: Vetor de termos independentes (numpy array).
        
    Returns:
        O vetor solução x (numpy array) ou None se a matriz for singular.
    """
    n = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    for i in range(n):
        # Pivoteamento Parcial
        # Encontra o maior elemento (em módulo) na coluna i a partir da linha i
        pivo_linha = i
        for k in range(i + 1, n):
            if abs(Ab[k, i]) > abs(Ab[pivo_linha, i]):
                pivo_linha = k
        
        # Troca a linha atual com a linha do pivô
        Ab[[i, pivo_linha]] = Ab[[pivo_linha, i]]

        # Verifica se a matriz é singular
        if Ab[i, i] == 0:
            return None  # Matriz singular

        # Eliminação
        for j in range(i + 1, n):
            fator = Ab[j, i] / Ab[i, i]
            Ab[j, i:] = Ab[j, i:] - fator * Ab[i, i:]

    # Retro-substituição
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        soma = np.dot(Ab[i, i+1:n], x[i+1:n])
        x[i] = (Ab[i, n] - soma) / Ab[i, i]

    return x
