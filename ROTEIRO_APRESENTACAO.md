# Roteiro de Apresentação — SuperMath

---

## 1. O que é o projeto

O SuperMath é uma **calculadora web de métodos numéricos** feita em Python com Streamlit.
A ideia foi criar uma ferramenta que resolve os algoritmos da disciplina de forma visual e interativa, sem precisar instalar nada — basta abrir no navegador.

> Link: https://supermath.streamlit.app/

**Stack usada:**
- **Python** — linguagem base
- **Streamlit** — framework que transforma scripts Python em apps web
- **NumPy / SciPy** — cálculos matriciais e numéricos
- **SymPy** — cálculo simbólico (usado para derivar funções automaticamente no Newton-Raphson)
- **Matplotlib** — geração dos gráficos
- **Groq API + Llama 3.3 70B** — modelo de linguagem que alimenta o assistente Clark Calc

---

## 2. Estrutura do sistema

Três módulos principais, acessados pelas abas no topo da tela:

| Módulo | O que faz |
|---|---|
| Raízes de Funções | Encontra o x onde f(x) = 0 |
| Sistemas Lineares | Resolve Ax = b |
| Ajuste de Curvas | Encontra a curva que melhor representa pontos experimentais |

Mais o **Clark Calc** — assistente de IA na barra lateral.

---

## 3. Módulo: Raízes de Funções

> **Ideia geral:** dado uma função f(x), queremos saber qual valor de x faz ela ser zero.

### Bisseção
- Pega um intervalo [a, b] onde a função troca de sinal (uma ponta positiva, outra negativa)
- Vai cortando o intervalo ao meio e ficando com o lado onde ainda tem a troca de sinal
- Continua até o intervalo ficar menor que a tolerância definida
- **Mais lento**, mas sempre converge se o intervalo for válido

### Ponto Fixo
- Reescreve f(x) = 0 como x = g(x) e fica repetindo x = g(x), g(g(x)), ...
- Cada resultado vira entrada da próxima iteração
- **O usuário precisa fornecer g(x)** — a escolha errada pode divergir
- Converge se |g'(x)| < 1 perto da raiz

### Newton-Raphson
- Usa a derivada da função para "mirar" melhor a raiz a cada passo
- Fórmula: `x_novo = x - f(x) / f'(x)`
- **O mais rápido** dos quatro (convergência quadrática)
- A derivada é calculada automaticamente pelo SymPy
- Falha se a derivada for zero no caminho

### Secantes
- Mesma ideia do Newton-Raphson, mas **sem precisar da derivada**
- Aproxima a derivada usando dois pontos: `f'(x) ≈ (f(x1) - f(x0)) / (x1 - x0)`
- Um pouco mais lento que o Newton, mas útil quando a derivada é difícil de obter
- Requer dois valores iniciais

---

## 4. Módulo: Sistemas Lineares

> **Ideia geral:** dado um sistema Ax = b, encontrar o vetor x.

### Eliminação de Gauss com Pivoteamento
- Transforma a matriz A em triangular superior usando operações de linha
- O **pivoteamento parcial** troca linhas para sempre usar o maior elemento disponível como divisor — evita erros numéricos por divisão por número muito pequeno
- Depois resolve por substituição retroativa (de baixo pra cima)
- **Método direto** — resolve em passos fixos, sem iterar

### Fatoração LU
- Decompõe A em duas matrizes: L (triangular inferior) e U (triangular superior)
- Resolve em duas etapas: primeiro Ly = b, depois Ux = y
- Vantagem: se precisar resolver o mesmo sistema com vários vetores b diferentes, a decomposição é feita só uma vez
- **Método direto**

### Jacobi
- **Método iterativo** — parte de um chute inicial e vai refinando
- Atualiza todas as variáveis ao mesmo tempo usando só os valores da iteração anterior
- Precisa que a matriz seja diagonalmente dominante para garantir convergência
- Mais simples de implementar, mas costuma precisar de mais iterações

### Gauss-Seidel
- Igual ao Jacobi, mas atualiza cada variável **na hora**, usando os valores mais recentes
- Geralmente converge mais rápido que o Jacobi com a mesma matriz
- Mesma condição de convergência (diagonal dominante)

---

## 5. Módulo: Ajuste de Curvas

> **Ideia geral:** dado um conjunto de pontos (x, y), encontrar a curva que passa mais perto de todos eles.

### Regressão Linear
- Ajusta uma reta `y = a·x + b` aos dados
- Minimiza a soma dos quadrados das distâncias entre os pontos e a reta (mínimos quadrados)
- Os coeficientes a e b são calculados diretamente por fórmula fechada

### Mínimos Quadrados Polinomial
- Mesma ideia, mas ajusta um polinômio de grau n: `y = c₀ + c₁x + c₂x² + ...`
- O usuário define o grau
- Grau maior = ajuste mais flexível, mas pode gerar overfitting com poucos pontos
- Usa `np.polyfit` internamente

---

## 6. Assistente Clark Calc

- IA integrada alimentada pelo **Llama 3.3 70B** via API do Groq (gratuita)
- Tem acesso em tempo real aos dados calculados na tela (via *function calling*)
- Pode ser acionado pelo botão "Como funciona?" ou por pergunta livre na barra lateral
- Explica o resultado atual, diagnostica erros e responde dúvidas sobre os métodos

> Dica na apresentação: execute um cálculo antes de acionar, para ele analisar o resultado específico.

---

## 7. O que pode ser perguntado — e as respostas

**"Por que usaram Streamlit?"**
> É o jeito mais rápido de colocar um script Python na web com interface gráfica, sem precisar de HTML/CSS/JS. Ideal para projetos acadêmicos e protótipos.

**"Por que o Newton-Raphson é mais rápido?"**
> Porque ele usa a derivada para corrigir o passo — a cada iteração o erro é aproximadamente elevado ao quadrado (convergência quadrática). Os outros métodos têm convergência linear.

**"Qual a diferença entre Jacobi e Gauss-Seidel?"**
> No Jacobi, você guarda todos os valores antigos e atualiza todo mundo de uma vez. No Gauss-Seidel, você já usa o valor novo de x₁ para calcular x₂, e assim por diante — aproveita a informação mais recente dentro da mesma iteração.

**"O que é pivoteamento parcial?"**
> É uma estratégia para evitar instabilidade numérica: antes de eliminar uma coluna, você troca linhas para garantir que o número que vai ser usado como divisor (o pivô) seja o maior possível naquela coluna. Evita divisão por números muito pequenos que amplificam erros de arredondamento.

**"O Clark Calc é um ChatGPT?"**
> Não exatamente. Usa o modelo Llama (open-source da Meta), rodando na infraestrutura do Groq. A diferença principal é que ele tem acesso aos dados da sua sessão via *function calling* — não é só um chat genérico, ele lê o que está na tela.

---

## Resumo de uma linha por método

| Método | Uma linha |
|---|---|
| Bisseção | Divide o intervalo ao meio até achar a raiz |
| Ponto Fixo | Itera x = g(x) até convergir |
| Newton-Raphson | Usa a derivada para convergir rápido |
| Secantes | Newton sem derivada — usa dois pontos |
| Gauss (pivoteamento) | Triangulariza o sistema e resolve de trás pra frente |
| Fatoração LU | Decompõe A = LU e resolve em duas etapas |
| Jacobi | Iterativo: atualiza tudo junto com valores velhos |
| Gauss-Seidel | Iterativo: atualiza usando os valores mais recentes |
| Regressão Linear | Ajusta a melhor reta aos dados |
| Mínimos Quadrados | Ajusta o melhor polinômio de grau n aos dados |
