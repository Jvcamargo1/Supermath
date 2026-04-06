import streamlit as st
import numpy as np
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import sympy

# --- Local Imports ---
from metodos_numericos.ajustes.regressao_linear import regressao_linear
from metodos_numericos.ajustes.minimos_quadrados import minimos_quadrados
from metodos_numericos.sistemas.gauss_pivoteamento import gauss_pivoteamento
from metodos_numericos.sistemas.fatoracao_lu import fatoracao_lu, solve_lu
from metodos_numericos.sistemas.jacobi import jacobi
from metodos_numericos.sistemas.gauss_seidel import gauss_seidel
from metodos_numericos.raizes.bissecao import bissecao
from metodos_numericos.raizes.ponto_fixo import ponto_fixo
from metodos_numericos.raizes.newton_raphson import newton_raphson
from metodos_numericos.raizes.secantes import secantes

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="SuperMath",
    page_icon="🧮",
    layout="wide",
    menu_items={
        'About': "### SuperMath\nCalculadora de computação numérica desenvolvida por um agente de IA."
    }
)

# Custom CSS for a more modern look
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    /* Style for containers with border */
    div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] > div[style*="border"] {
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    /* Primary button style */
    div.stButton > button[kind="primary"] {
        background-color: #0068C9;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Função principal da aplicação Streamlit com navegação por abas."""
    st.title("🧮 SuperMath")

    # --- BARRA LATERAL DO CHATBOT ---
    # Movido para a barra lateral para não obstruir a calculadora.
    with st.sidebar:
        st.header("🤖 Assistente IA")
        render_chatbot()

    tab_raizes, tab_sistemas, tab_ajustes = st.tabs([
        "🎯 Raízes de Funções",
        "🔢 Sistemas Lineares",
        "📈 Ajuste de Curvas"
    ])

    with tab_raizes:
        show_raizes_page()
    with tab_sistemas:
        show_sistemas_page()
    with tab_ajustes:
        show_ajustes_page()


def parse_function(func_str, var_symbol='x'):
    """Converte uma string de função em uma função numérica usando Sympy."""
    try:
        x = sympy.Symbol(var_symbol)
        local_dict = {"exp": sympy.exp, "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan, "sqrt": sympy.sqrt, "log": sympy.log}
        expr = sympy.sympify(func_str, locals=local_dict)
        return sympy.lambdify(x, expr, 'numpy'), expr, None
    except (sympy.SympifyError, SyntaxError) as e:
        return None, None, f"Erro ao interpretar a função: '{func_str}'. Verifique a sintaxe. ({e})"


def show_raizes_page():
    """Exibe a página para os métodos de raízes de funções."""
    st.subheader("Encontre a raiz de `f(x) = 0` para uma dada função")
    
    escolha_metodo = st.selectbox("Escolha o método:", ["Bisseção", "Ponto Fixo", "Newton-Raphson", "Secantes"])
    
    with st.container(border=True):
        func_str = st.text_input("Função f(x)", value="x**3 - x - 2", help="Use sintaxe Python. Funções comuns: exp, sin, cos, tan, sqrt, log.")
        g_func_str = st.text_input("Função de iteração g(x)", value="(x + 2)**(1/3)", help="Apenas para Ponto Fixo: forneça g(x) tal que g(x)=x na raiz.") if escolha_metodo == "Ponto Fixo" else None

        params = {}
        cols = st.columns(4)
        if escolha_metodo == "Bisseção":
            params['a'] = cols[0].number_input("Intervalo (a)", value=1.0, format="%.4f")
            params['b'] = cols[1].number_input("Intervalo (b)", value=2.0, format="%.4f")
        elif escolha_metodo in ["Ponto Fixo", "Newton-Raphson"]:
            params['x0'] = cols[0].number_input("Chute inicial (x0)", value=1.0, format="%.4f")
        elif escolha_metodo == "Secantes":
            params['x0'] = cols[0].number_input("Chute (x0)", value=1.0, format="%.4f")
            params['x1'] = cols[1].number_input("Chute (x1)", value=2.0, format="%.4f")

        params['tol'] = cols[2].number_input("Tolerância", value=1e-6, format="%.2e")
        params['max_iter'] = cols[3].number_input("Max. Iterações", value=100, min_value=1, step=1)

        if st.button("Encontrar Raiz", type="primary", key="raiz_btn"):
            func, expr, err = parse_function(func_str)
            if err: st.error(err); return

            resultado, info = None, ""
            try:
                if escolha_metodo == "Bisseção": resultado, info = bissecao(func, params['a'], params['b'], params['tol'], params['max_iter'])
                elif escolha_metodo == "Ponto Fixo":
                    if not g_func_str: st.error("Por favor, forneça a função de iteração g(x)."); return
                    g_func, _, err_g = parse_function(g_func_str); 
                    if err_g: st.error(f"Erro na função g(x): {err_g}"); return
                    resultado, info = ponto_fixo(g_func, params['x0'], params['tol'], params['max_iter'])
                elif escolha_metodo == "Newton-Raphson":
                    df_expr = sympy.diff(expr, sympy.Symbol('x')); df, _, _ = parse_function(str(df_expr))
                    with st.expander("Derivada `f'(x)` calculada"): st.latex(sympy.latex(df_expr))
                    resultado, info = newton_raphson(func, df, params['x0'], params['tol'], params['max_iter'])
                elif escolha_metodo == "Secantes": resultado, info = secantes(func, params['x0'], params['x1'], params['tol'], params['max_iter'])
                
                # --- NEW COMPACT LAYOUT ---
                if isinstance(resultado, (int, float)):
                    res_col, plot_col = st.columns([1, 2])
                    with res_col:
                        st.success(f"**Raiz encontrada:**\n## {resultado:.7f}")
                        st.info(f"**Detalhes:**\n\n{info}")
                    with plot_col:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        plot_range = np.linspace(resultado - 2, resultado + 2, 400)
                        y_vals = func(plot_range)
                        ax.plot(plot_range, y_vals, label=f'f(x) = {func_str}')
                        ax.axhline(0, color='gray', lw=0.7)
                        ax.axvline(resultado, color='red', ls='--', label=f'Raiz ≈ {resultado:.4f}')
                        ax.scatter(resultado, func(resultado), color='red', zorder=5)
                        ax.set_xlabel("x"); ax.set_ylabel("f(x)"); ax.set_title("Gráfico da Função e Raiz"); ax.legend(); ax.grid(True)
                        st.pyplot(fig, use_container_width=True)
                else: 
                    st.error(f"Não foi possível encontrar a raiz. Motivo: {info}")
            except Exception as e: 
                st.error(f"Ocorreu um erro de cálculo: {e}")

def show_sistemas_page():
    """Exibe a página para os métodos de sistemas lineares."""
    st.subheader("Resolva um sistema de equações lineares no formato `Ax = b`")
    escolha_metodo = st.selectbox("Escolha o método:", ["Eliminação de Gauss com Pivoteamento", "Fatoração LU", "Jacobi", "Gauss-Seidel"])

    with st.container(border=True):
        a_str = st.text_area("Matriz A", "4, -1, 1\n-1, 4, -2\n1, -2, 4", height=120, help="Separe linhas com 'enter' e elementos com vírgula.")
        b_str = st.text_area("Vetor b", "12, -1, 5", height=50, help="Separe elementos com vírgula.")

        if escolha_metodo in ["Jacobi", "Gauss-Seidel"]:
            cols_iter = st.columns(3)
            x0_str = cols_iter[0].text_input("Chute inicial x0", "0, 0, 0")
            tol = cols_iter[1].number_input("Tolerância", value=1e-6, format="%.2e")
            max_iter = cols_iter[2].number_input("Max. Iterações", value=100, min_value=1, step=1)
        
        if st.button("Calcular Solução", type="primary", key="sis_btn"):
            try:
                mat_A = np.array([list(map(float, row.split(','))) for row in a_str.strip().split('\n')]); vet_b = np.array(list(map(float, b_str.strip().split(','))))
                if mat_A.shape[0] != mat_A.shape[1]: st.error("A matriz A deve ser quadrada."); return
                if mat_A.shape[0] != len(vet_b): st.error("As dimensões de A e b são incompatíveis."); return
                
                if escolha_metodo == "Eliminação de Gauss com Pivoteamento":
                    solucao = gauss_pivoteamento(mat_A, vet_b)
                    if solucao is not None: st.success(f"**Vetor solução x:** `{np.array2string(solucao, precision=6)}`")
                    else: st.error("A matriz é singular. Não há solução única.")
                elif escolha_metodo == "Fatoração LU":
                    P, L, U = fatoracao_lu(mat_A)
                    if P is not None:
                        solucao = solve_lu(P, L, U, vet_b)
                        st.success(f"**Vetor solução x:** `{np.array2string(solucao, precision=6)}`")
                        with st.expander("Ver Matrizes Decompostas (P, L, U)"):
                            st.text("Matriz de Permutação (P):"); st.code(np.array2string(P, precision=4), language=None)
                            st.text("Matriz Triangular Inferior (L):"); st.code(np.array2string(L, precision=4), language=None)
                            st.text("Matriz Triangular Superior (U):"); st.code(np.array2string(U, precision=4), language=None)
                    else: st.error("A fatoração LU falhou. A matriz pode ser singular.")
                elif escolha_metodo in ["Jacobi", "Gauss-Seidel"]:
                    vet_x0 = np.array(list(map(float, x0_str.split(','))))
                    if len(vet_x0) != len(vet_b): st.error("O vetor de chute inicial x0 tem dimensões incorretas."); return
                    solucao, k = jacobi(mat_A, vet_b, vet_x0, tol, max_iter) if escolha_metodo == "Jacobi" else gauss_seidel(mat_A, vet_b, vet_x0, tol, max_iter)
                    st.success(f"**Vetor solução x:** `{np.array2string(solucao, precision=6)}`")
                    st.info(f"Solução encontrada em {k} iterações.")
            except ValueError: st.error("Erro nos dados de entrada. Verifique a formatação.")
            except Exception as e: st.error(f"Ocorreu um erro de cálculo: {e}")

def show_ajustes_page():
    """Exibe a página para os métodos de ajuste de curvas."""
    st.subheader("Ajuste uma curva a um conjunto de pontos de dados (x, y)")
    escolha_metodo = st.selectbox("Escolha o método:", ["Regressão Linear", "Mínimos Quadrados (Polinomial)"])

    with st.container(border=True):
        cols = st.columns(2)
        x_str = cols[0].text_area("Valores de X", "1, 2, 3, 4, 5, 6, 7", help="Separe os números por vírgula.")
        y_str = cols[1].text_area("Valores de Y", "1.5, 3.8, 6.7, 8.5, 11.2, 13.5, 16.0", help="Separe os números por vírgula.")

        grau = st.number_input("Grau do Polinômio:", min_value=1, max_value=10, value=2, step=1) if "Mínimos Quadrados" in escolha_metodo else 1

        if st.button("Calcular Ajuste", type="primary", key="ajuste_btn"):
            try:
                x_data = np.array([float(i.strip()) for i in x_str.split(',')]); y_data = np.array([float(i.strip()) for i in y_str.split(',')])
                if len(x_data) != len(y_data): st.error("Os conjuntos X e Y devem ter o mesmo número de pontos."); return
                if len(x_data) < 2: st.error("São necessários pelo menos 2 pontos para o ajuste."); return
                
                res_col, plot_col = st.columns([1, 2])

                with res_col:
                    if escolha_metodo == "Regressão Linear":
                        a, b = regressao_linear(x_data, y_data)
                        if a is None: st.error("Não foi possível calcular a regressão."); return
                        st.success("Equação da Reta Ajustada:")
                        st.latex(f"y = {a:.5f}x {'+' if b >= 0 else '-'} {abs(b):.5f}")
                    else: # Mínimos Quadrados
                        if len(x_data) <= grau: st.error("O número de pontos deve ser maior que o grau do polinômio."); return
                        coefs = minimos_quadrados(x_data, y_data, grau)
                        if coefs is None: st.error("Não foi possível calcular o ajuste polinomial."); return
                        p = np.poly1d(coefs)
                        eq = "y = " + " ".join([f"{'' if i==0 else ('+ ' if c>=0 else '- ')}{abs(c):.4f}x^{grau-i}" for i, c in enumerate(coefs)]).replace("x^0", "").replace("x^1 ", "x ")
                        st.success("Equação do Polinômio Ajustado:")
                        st.latex(eq)
                
                with plot_col:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.scatter(x_data, y_data, label='Dados Originais')
                    x_fit = np.linspace(x_data.min(), x_data.max(), 400)
                    if escolha_metodo == "Regressão Linear":
                        a, b = regressao_linear(x_data, y_data) # Recalculate to have vars in scope
                        y_fit = a * x_fit + b
                    else:
                        coefs = minimos_quadrados(x_data, y_data, grau) # Recalculate to have vars in scope
                        p = np.poly1d(coefs)
                        y_fit = p(x_fit)
                    ax.plot(x_fit, y_fit, color='red', label='Curva Ajustada')
                    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_title("Ajuste de Curva vs Dados Originais"); ax.legend(); ax.grid(True)
                    st.pyplot(fig, use_container_width=True)

            except ValueError: st.error("Erro nos dados de entrada. Verifique a formatação.")
            except Exception as e: st.error(f"Ocorreu um erro inesperado: {e}")


def render_chatbot():
    """Renderiza a interface do Chatbot na barra lateral."""
    st.markdown("Tire suas dúvidas sobre os métodos numéricos, algoritmos ou peça ajuda para entender as respostas!")

    # Tenta carregar das Secrets do Streamlit Cloud; se não existir, usa as variáveis de ambiente local (.env)
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.warning("Chave da API do Groq não encontrada. Verifique seu `.env` ou os Secrets do Streamlit.")
        st.code("GROQ_API_KEY='sua_chave_aqui'", language="bash")
        return

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
    except ImportError:
        st.error("A biblioteca 'groq' não está instalada. Execute `pip install groq` no seu terminal.")
        return
    except Exception as e:
        st.error(f"Erro ao inicializar o cliente Groq: {e}")
        return

    # Inicializa o histórico do chat na sessão do Streamlit
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "Você é o assistente virtual do SuperMath, uma calculadora de cálculo numérico para estudantes de Ciência da Computação. Seu objetivo é ajudar o usuário a usar o site e explicar de forma didática e fácil como funcionam métodos como Bisseção, Newton-Raphson, Eliminação de Gauss, Ajuste de Curvas, etc. Responda em português de forma clara e amigável."}
        ]

    # Exibe as mensagens do chat (ignora a mensagem de sistema)
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Campo de entrada para a pergunta do usuário
    if prompt := st.chat_input("Faça uma pergunta (ex: 'Como funciona o método de Newton-Raphson?')..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                # Chamada para a API do Groq usando o modelo Llama 3.1 veloz
                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=st.session_state.messages,
                    temperature=0.3,
                    max_tokens=1024,
                )
                response = completion.choices[0].message.content
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error("Ocorreu um erro na comunicação com o Groq. Isso pode acontecer se a API retornar uma mensagem com caracteres especiais.")
                # st.exception é a forma mais robusta de exibir um erro, pois lida com a formatação e evita problemas de encoding.
                st.exception(e)


if __name__ == "__main__":
    main()
