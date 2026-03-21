import numpy as np

# Funções de Bisseção
def bissecao(func, a, b, tol=1e-6, max_iter=100):
    """
    Encontra a raiz de uma função usando o método da Bisseção.
    
    Args:
        func: A função (numérica, já "lambdified") para a qual encontrar a raiz.
        a: Início do intervalo.
        b: Fim do intervalo.
        tol: Tolerância (critério de parada).
        max_iter: Número máximo de iterações.
        
    Returns:
        A raiz aproximada, número de iterações, ou uma mensagem de erro.
    """
    if func(a) * func(b) >= 0:
        return None, "A função não muda de sinal no intervalo [a, b]. O Teorema de Bolzano não pode ser garantido."

    k = 0
    while (b - a) / 2 > tol and k < max_iter:
        p = (a + b) / 2
        if func(p) == 0:
            return p, k + 1
        elif func(a) * func(p) < 0:
            b = p
        else:
            a = p
        k += 1
        
    return (a + b) / 2, k
