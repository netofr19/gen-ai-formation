import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
print("Chave carregada:", api_key[:5] + "...")

llm = ChatOpenAI(
    model="meta-llama/llama-3.1-8b-instruct",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

lista_perguntas = [
    "Quero visitar um lugar no Brasil, famoso por praias e cultura. Pode sugerir?",
    "Qual a melhor época do ano para ir?"
]

for pergunta in lista_perguntas:
    response = llm.invoke(pergunta)
    print(f"Usuário: {pergunta}")
    print(f"IA: {response.content}")
    print(f"\n{'-'*60}\n")
