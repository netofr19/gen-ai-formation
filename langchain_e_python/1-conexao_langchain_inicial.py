from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
print("Chave carregada:", api_key[:5] + "...")

llm = ChatOpenAI(
    model="meta-llama/llama-3.1-8b-instruct",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

numero_dias = 7
numero_criancas = 2
atividade = "praia"

prompt_template = PromptTemplate(
    template="""
    Quantas copas do mundo de futebol ganhou a seleção Brasileira? Explique um pouco sobre cada copa do mundo ganha.
    """
)

prompt = prompt_template.format(
    numero_dias=numero_dias,
    numero_criancas=numero_criancas,
    atividade=atividade
)

print(f"====== PROMPT ======\n{prompt}")


response = llm.invoke(prompt)
print(f"====== RESPONSE ======")
print(response.content)
