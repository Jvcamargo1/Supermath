import numpy as np

# Funções de Regressão Linear
def regressao_linear(x, y):
    """
    Ajusta uma reta (y = a*x + b) a um conjunto de pontos (x, y).
    
    Args:
        x: Lista ou array de valores de x.
        y: Lista ou array de valores de y.
        
    Returns:
        Os coeficientes 'a' (inclinação) e 'b' (intercepto) da reta ajustada.
    """
    x = np.array(x)
    y = np.array(y)
    
    n = len(x)
    
    # Soma dos produtos
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x_sq = np.sum(x**2)
    
    # Cálculo dos coeficientes 'a' e 'b'
    # a = (n * sum(xy) - sum(x) * sum(y)) / (n * sum(x^2) - (sum(x))^2)
    # b = mean(y) - a * mean(x)
    
    try:
        a = (n * sum_xy - sum_x * sum_y) / (n * sum_x_sq - sum_x**2)
        b = (sum_y / n) - a * (sum_x / n)
    except ZeroDivisionError:
        return None, None # Retorna None se a divisão por zero ocorrer

    return a, b
