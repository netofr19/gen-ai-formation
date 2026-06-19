from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import Field, BaseModel
from dotenv import load_dotenv

from langchain_core.globals import set_debug

import os

load_dotenv()
set_debug(True)

api_key = os.getenv("OPENROUTER_API_KEY")
print("Chave carregada:", api_key[:5] + "...")

class Destino(BaseModel):
    cidade: str = Field(description="Cidade sugerida para o interesse dado")
    motivo: str = Field(description="Motivo da sugestão da cidade, de forma extensiva")

llm = ChatOpenAI(
    model="meta-llama/llama-3.1-8b-instruct",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

parser = JsonOutputParser(pydantic_object=Destino)

prompt_template = PromptTemplate(
    template="""
    Sugiro uma cidade dado o meu interesse por {interesse}.
    {output_format}
    """,
    input_variables=["interesse"],
    partial_variables={"output_format": parser.get_format_instructions()}
)

chain = prompt_template | llm | parser


response = chain.invoke(
    {'interesse': "praias"}
)

print(response)
