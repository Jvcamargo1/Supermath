import numpy as np

# Funções de Jacobi
def jacobi(A, b, x0, tol=1e-10, max_iter=100):
    """
    Resolve um sistema linear Ax=b usando o método iterativo de Jacobi.
    
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
        x_novo = np.zeros(n)
        for i in range(n):
            soma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x_novo[i] = (b[i] - soma) / A[i, i]
        
        # Critério de parada: erro relativo
        if np.linalg.norm(x_novo - x, ord=np.inf) / (np.linalg.norm(x_novo, ord=np.inf) + 1e-12) < tol:
            return x_novo, k + 1
            
        x = x_novo

    return x, max_iter # Retorna a última iteração se max_iter for atingido
