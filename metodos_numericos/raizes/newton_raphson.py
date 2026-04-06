import numpy as np

# Funções de Newton-Raphson
def newton_raphson(func, df, x0, tol=1e-6, max_iter=100):
    """
    Encontra a raiz de uma função usando o método de Newton-Raphson.
    
    Args:
        func: A função f(x) (numérica, já "lambdified").
        df: A derivada da função f'(x) (numérica, já "lambdified").
        x0: Chute inicial.
        tol: Tolerância (critério de parada).
        max_iter: Número máximo de iterações.
        
    Returns:
        A raiz aproximada, número de iterações, ou uma mensagem de erro.
    """
    x = x0
    for k in range(max_iter):
        fx = func(x)
        dfx = df(x)

        if abs(dfx) < 1e-12: # Evita divisão por um número muito pequeno
            return None, "A derivada se aproximou de zero. O método falhou."

        try:
            x_novo = x - fx / dfx
        except (ZeroDivisionError, OverflowError):
            return None, "Ocorreu um erro numérico (divisão por zero ou overflow). O método falhou."

        if abs(x_novo - x) < tol:
            return x_novo, k + 1
        
        x = x_novo

    return None, f"O método não convergiu após {max_iter} iterações."
