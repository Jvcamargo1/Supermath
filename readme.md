# SuperMath - Calculadora de Computação Numérica

O **SuperMath** é uma aplicação web desenvolvida em Python com o framework Streamlit. O objetivo do projeto é fornecer uma interface para resolução, análise e experimentação de problemas e algoritmos clássicos de cálculo numérico de acordo com o que foi especificado no projeto da aula de Computação Numérica do curso de Ciência da Computação da UNISAGRADO.

---

## ✨ Funcionalidades

- **🎯 Raízes de Funções:** Bisseção, Ponto Fixo, Newton-Raphson e Secantes.
- **🔢 Sistemas Lineares:** Eliminação de Gauss com Pivoteamento, Fatoração LU, Jacobi e Gauss-Seidel.
- **📈 Ajuste de Curvas:** Regressão Linear e Mínimos Quadrados (Polinomial).
- **🤖 Assistente IA Contextual:** Chatbot inteligente alimentado pela API do Groq (modelo Llama 3.3 70B). O assistente possui *Function Calling* e **lê os dados da sua tela em tempo real**, sendo capaz de explicar o motivo exato de uma falha matemática no seu cálculo ou detalhar passo a passo como o resultado atual foi encontrado!

---

## Como Acessar o projeto

O app foi upado no StreamLit Cloud Community para facilitar o acesso: https://supermath.streamlit.app/

---

## ☁️ Executando com GitHub Codespaces (Sem instalar nada)

Como o projeto possui um contêiner de desenvolvimento configurado (`.devcontainer`), você pode rodar a aplicação com todas as dependências instaladas diretamente pelo navegador:

1. Na página inicial do repositório no GitHub, clique no botão verde **"<> Code"**.
2. Selecione a aba **"Codespaces"** e clique em **"Create codespace on main"**.
3. Aguarde o ambiente carregar. Ele executará o `requirements.txt` automaticamente e abrirá o aplicativo em uma nova aba (porta 8501).

*Nota: Para usar o Chatbot com IA no Codespaces, você precisará adicionar a sua `GROQ_API_KEY` criando o arquivo `.env` dentro do ambiente.*

---

## 💻 Como Executar o Projeto Localmente

Siga as instruções abaixo para configurar o ambiente e executar a aplicação em sua máquina local.

---

## 1. Pré-requisitos

Certifique-se de ter os seguintes itens instalados:

- Python (versão 3.8 ou superior)
- Git

---

## 2. Clonando o Repositório

Clone o repositório para o seu ambiente local:

```bash
git clone https://github.com/Jvcamargo1/Supermath.git
cd Supermath
```

---

## 3. Configurando o Ambiente Virtual

Recomenda-se o uso de um ambiente virtual para isolar as dependências do projeto:

```bash
# Criação do ambiente virtual
python -m venv venv

# Ativação no Windows
venv\Scripts\activate

# Ativação no Linux/macOS
source venv/bin/activate
```

---

## 4. Instalando as Dependências

Com o ambiente virtual ativado, instale as bibliotecas necessárias:

```bash
pip install -r requirements.txt
```

---

## 5. Configurando o Assistente IA (Chave da API)

Para que o chatbot de Inteligência Artificial funcione localmente, você precisará configurar uma chave de API gratuita do Groq:

1. Crie uma conta e gere sua chave em: [Groq Console](https://console.groq.com/keys)
2. Na raiz do projeto, faça uma cópia do arquivo `.env.example` e renomeie essa cópia para `.env`
3. Abra o arquivo `.env` recém-criado e substitua o valor de exemplo pela sua chave real:

```env
GROQ_API_KEY=sua_chave_api_real_aqui
```

---

## 6. Executando a Aplicação

Inicie o servidor local do Streamlit executando o comando abaixo na raiz do projeto:

```bash
streamlit run app.py
```

A aplicação estará disponível no navegador no endereço:

http://localhost:8501

---
