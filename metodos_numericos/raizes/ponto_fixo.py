import numpy as np

# Funções de Ponto Fixo
def ponto_fixo(g_func, x0, tol=1e-6, max_iter=100):
    """
    Encontra a raiz de uma função usando o método do Ponto Fixo.
    
    Args:
        g_func: A função de iteração g(x) (numérica, já "lambdified").
        x0: Chute inicial.
        tol: Tolerância (critério de parada).
        max_iter: Número máximo de iterações.
        
    Returns:
        A raiz aproximada, número de iterações, ou uma mensagem de erro.
    """
    x = x0
    for k in range(max_iter):
        try:
            x_novo = g_func(x)
        except OverflowError:
            return None, "Ocorreu um overflow. O método pode ter divergido."

        if abs(x_novo - x) < tol:
            return x_novo, k + 1
        x = x_novo

    return None, f"O método não convergiu após {max_iter} iterações."
