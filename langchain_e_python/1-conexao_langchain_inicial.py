from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
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
    Crie um roteiro de viagem de {numero_dias} dias, 
    para uma família com {numero_criancas} crianças, 
    que gosta de {atividade}
    """
)

prompt = prompt_template.format(
    numero_dias=numero_dias,
    numero_criancas=numero_criancas,
    atividade=atividade
)

print(f"====== PROMPT ======\n{prompt}")


response = llm.invoke(prompt)
console = Console()
console.print(Markdown(response.content))
