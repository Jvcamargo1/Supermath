import numpy as np

# Funções de Mínimos Quadrados
def minimos_quadrados(x, y, grau):
    """
    Ajusta um polinômio de um dado grau a um conjunto de pontos (x, y)
    usando o método dos mínimos quadrados.
    
    Args:
        x: Lista ou array de valores de x.
        y: Lista ou array de valores de y.
        grau: O grau do polinômio a ser ajustado.
        
    Returns:
        Os coeficientes do polinômio ajustado (do maior grau para o menor).
        Retorna None se o ajuste falhar.
    """
    try:
        # np.polyfit retorna os coeficientes do polinômio que minimiza o erro quadrático.
        # Por exemplo, para grau 2, retorna [c2, c1, c0] para c2*x^2 + c1*x + c0.
        coeficientes = np.polyfit(x, y, grau)
        return coeficientes
    except (np.linalg.LinAlgError, ValueError):
        # Captura erros comuns do polyfit, como matriz singular ou dados inválidos.
        return None
