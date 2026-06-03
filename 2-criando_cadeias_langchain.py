from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

prompt_template = PromptTemplate(
    template="""
    Sugiro uma cidade dado o meu interesse por {interesse}.
    """,
    input_variables=["interesse"]
)

chain = prompt_template | llm | StrOutputParser()


response = chain.invoke(
    {'interesse': "praias"}
)

console = Console()
console.print(Markdown(response))
