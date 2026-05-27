# SuperMath — Rotinas Numéricas

Scripts das funções implementadas no projeto, organizadas por módulo.

---

## Raízes de Funções

### Bisseção
`metodos_numericos/raizes/bissecao.py`

```python
def bissecao(func, a, b, tol=1e-6, max_iter=100):
    if func(a) * func(b) >= 0:
        return None, "A função não muda de sinal no intervalo [a, b]."

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
```

---

### Ponto Fixo
`metodos_numericos/raizes/ponto_fixo.py`

```python
def ponto_fixo(g_func, x0, tol=1e-6, max_iter=100):
    x = x0
    for k in range(max_iter):
        try:
            x_novo = g_func(x)
        except OverflowError:
            return None, "Overflow — o método divergiu."

        if abs(x_novo - x) < tol:
            return x_novo, k + 1
        x = x_novo

    return None, f"Não convergiu após {max_iter} iterações."
```

---

### Newton-Raphson
`metodos_numericos/raizes/newton_raphson.py`

```python
def newton_raphson(func, df, x0, tol=1e-6, max_iter=100):
    x = x0
    for k in range(max_iter):
        fx  = func(x)
        dfx = df(x)

        if abs(dfx) < 1e-12:
            return None, "Derivada próxima de zero — método falhou."

        try:
            x_novo = x - fx / dfx
        except (ZeroDivisionError, OverflowError):
            return None, "Erro numérico — método falhou."

        if abs(x_novo - x) < tol:
            return x_novo, k + 1
        x = x_novo

    return None, f"Não convergiu após {max_iter} iterações."
```

> A derivada `df` é calculada automaticamente pelo sistema via SymPy.

---

### Secantes
`metodos_numericos/raizes/secantes.py`

```python
def secantes(func, x0, x1, tol=1e-6, max_iter=100):
    fx0 = func(x0)
    fx1 = func(x1)

    for k in range(max_iter):
        if abs(fx1 - fx0) < 1e-12:
            return None, "f(x1) - f(x0) próximo de zero — método falhou."

        try:
            x_novo = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        except (ZeroDivisionError, OverflowError):
            return None, "Erro numérico — método falhou."

        if abs(x_novo - x1) < tol:
            return x_novo, k + 1

        x0, x1 = x1, x_novo
        fx0, fx1 = fx1, func(x_novo)

    return None, f"Não convergiu após {max_iter} iterações."
```

---

## Sistemas Lineares

### Eliminação de Gauss com Pivoteamento
`metodos_numericos/sistemas/gauss_pivoteamento.py`

```python
def gauss_pivoteamento(A, b):
    n  = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    for i in range(n):
        # Pivoteamento parcial
        pivo = max(range(i, n), key=lambda k: abs(Ab[k, i]))
        Ab[[i, pivo]] = Ab[[pivo, i]]

        if Ab[i, i] == 0:
            return None  # Matriz singular

        for j in range(i + 1, n):
            fator     = Ab[j, i] / Ab[i, i]
            Ab[j, i:] = Ab[j, i:] - fator * Ab[i, i:]

    # Retro-substituição
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, n] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]

    return x
```

---

### Fatoração LU
`metodos_numericos/sistemas/fatoracao_lu.py`

```python
from scipy.linalg import lu as scipy_lu, solve_triangular

def fatoracao_lu(A):
    try:
        P, L, U = scipy_lu(A)
        return P, L, U
    except (np.linalg.LinAlgError, ValueError):
        return None, None, None

def solve_lu(P, L, U, b):
    Pb = np.dot(P, b)
    y  = solve_triangular(L, Pb, lower=True)   # Ly  = Pb
    x  = solve_triangular(U, y,  lower=False)  # Ux  = y
    return x
```

---

### Jacobi
`metodos_numericos/sistemas/jacobi.py`

```python
def jacobi(A, b, x0, tol=1e-10, max_iter=100):
    n = len(A)
    x = x0.copy()

    for k in range(max_iter):
        x_novo = np.zeros(n)
        for i in range(n):
            soma      = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x_novo[i] = (b[i] - soma) / A[i, i]

        if np.linalg.norm(x_novo - x, ord=np.inf) / (np.linalg.norm(x_novo, ord=np.inf) + 1e-12) < tol:
            return x_novo, k + 1
        x = x_novo

    return x, max_iter
```

---

### Gauss-Seidel
`metodos_numericos/sistemas/gauss_seidel.py`

```python
def gauss_seidel(A, b, x0, tol=1e-10, max_iter=100):
    n = len(A)
    x = x0.copy()

    for k in range(max_iter):
        x_ant = x.copy()
        for i in range(n):
            soma1 = np.dot(A[i, :i],   x[:i])
            soma2 = np.dot(A[i, i+1:], x_ant[i+1:])
            x[i]  = (b[i] - soma1 - soma2) / A[i, i]

        if np.linalg.norm(x - x_ant, ord=np.inf) / (np.linalg.norm(x, ord=np.inf) + 1e-12) < tol:
            return x, k + 1

    return x, max_iter
```

---

## Ajuste de Curvas

### Regressão Linear
`metodos_numericos/ajustes/regressao_linear.py`

```python
def regressao_linear(x, y):
    x, y = np.array(x), np.array(y)
    n       = len(x)
    sum_x   = np.sum(x)
    sum_y   = np.sum(y)
    sum_xy  = np.sum(x * y)
    sum_x2  = np.sum(x ** 2)

    try:
        a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        b = (sum_y / n) - a * (sum_x / n)
    except ZeroDivisionError:
        return None, None

    return a, b   # y = a*x + b
```

---

### Mínimos Quadrados Polinomial
`metodos_numericos/ajustes/minimos_quadrados.py`

```python
def minimos_quadrados(x, y, grau):
    try:
        coeficientes = np.polyfit(x, y, grau)
        return coeficientes  # [c_n, ..., c_1, c_0]
    except (np.linalg.LinAlgError, ValueError):
        return None
```

> Retorna coeficientes do maior para o menor grau, compatíveis com `np.poly1d`.
