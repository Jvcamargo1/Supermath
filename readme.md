# SuperMath - Calculadora de Computação Numérica

O **SuperMath** é uma aplicação web desenvolvida em Python com o framework Streamlit. O objetivo do projeto é fornecer uma interface para resolução, análise e experimentação de problemas e algoritmos clássicos de cálculo numérico de acordo com o que foi especificado no projeto da aula de Computação Numérica do curso de Ciência da Computação da UNISAGRADO.

---

## ✨ Funcionalidades

- **🎯 Raízes de Funções:** Bisseção, Ponto Fixo, Newton-Raphson e Secantes.
- **🔢 Sistemas Lineares:** Eliminação de Gauss com Pivoteamento, Fatoração LU, Jacobi e Gauss-Seidel.
- **📈 Ajuste de Curvas:** Regressão Linear e Mínimos Quadrados (Polinomial).
- **🤖 Assistente IA Integrado:** Chatbot inteligente alimentado pela API do Groq (modelo Llama 3) para tirar dúvidas sobre os métodos numéricos e ajudar na interpretação dos resultados!

---

## Como Acessar o projeto

O app foi upado no StreamLit Cloud Community para facilitar o acesso: https://trabalhocompnum.streamlit.app/

Caso queira executar localmente, siga as proximas instruções

---

## Como Executar o Projeto

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
git clone https://github.com/Jvcamargo1/comp_num.git
cd comp_num
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
