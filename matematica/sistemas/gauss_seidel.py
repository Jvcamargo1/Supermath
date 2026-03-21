import numpy as np

# Funções de Gauss-Seidel
def gauss_seidel(A, b, x0, tol=1e-10, max_iter=100):
    """
    Resolve um sistema linear Ax=b usando o método iterativo de Gauss-Seidel.
    
    Args:
        A: Matriz de coeficientes.
        b: Vetor de termos independentes.
        x0: Chute inicial.
        tol: Tolerância (critério de parada).
        max_iter: Número máximo de iterações.
        
    Returns:
        O vetor solução aproximado x, o número de iterações, ou uma mensagem de erro.
    """
    n = len(A)
    x = x0.copy()

    # Verifica o critério de convergência (diagonal dominante)
    diag = np.diag(np.abs(A))
    soma_abs_nao_diag = np.sum(np.abs(A), axis=1) - diag
    if not np.all(diag > soma_abs_nao_diag):
        # Aviso, não é um erro fatal, o método pode convergir mesmo assim.
        print("Aviso: A matriz não é estritamente diagonal dominante. A convergência não é garantida.")

    for k in range(max_iter):
        x_anterior = x.copy()
        for i in range(n):
            soma1 = np.dot(A[i, :i], x[:i])
            soma2 = np.dot(A[i, i+1:], x_anterior[i+1:])
            x[i] = (b[i] - soma1 - soma2) / A[i, i]
            
        # Critério de parada: erro relativo
        if np.linalg.norm(x - x_anterior, ord=np.inf) / (np.linalg.norm(x, ord=np.inf) + 1e-12) < tol:
            return x, k + 1

    return x, max_iter # Retorna a última iteração se max_iter for atingido
