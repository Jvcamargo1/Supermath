import numpy as np

# Funções do método das Secantes
def secantes(func, x0, x1, tol=1e-6, max_iter=100):
    """
    Encontra a raiz de uma função usando o método das Secantes.
    
    Args:
        func: A função f(x) (numérica, já "lambdified").
        x0: Primeiro chute inicial.
        x1: Segundo chute inicial.
        tol: Tolerância (critério de parada).
        max_iter: Número máximo de iterações.
        
    Returns:
        A raiz aproximada, número de iterações, ou uma mensagem de erro.
    """
    fx0 = func(x0)
    fx1 = func(x1)

    for k in range(max_iter):
        if abs(fx1 - fx0) < 1e-12:
            return None, "A diferença f(x1) - f(x0) se aproximou de zero. O método falhou."
        
        try:
            x_novo = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        except (ZeroDivisionError, OverflowError):
             return None, "Ocorreu um erro numérico (divisão por zero ou overflow). O método falhou."

        if abs(x_novo - x1) < tol:
            return x_novo, k + 1
        
        x0, x1 = x1, x_novo
        fx0, fx1 = fx1, func(x_novo)

    return None, f"O método não convergiu após {max_iter} iterações."
