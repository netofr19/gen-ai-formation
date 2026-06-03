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

class Restaurante(BaseModel):
    cidade: str = Field(description="Cidade sugerida para o interesse dado")
    restaurantes: str = Field("Resurantes recomendados na cidade")

llm = ChatOpenAI(
    model="meta-llama/llama-3.1-8b-instruct",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

parser_destino = JsonOutputParser(pydantic_object=Destino)
parser_restaurante = JsonOutputParser(pydantic_object=Restaurante)

prompt_template_destino = PromptTemplate(
    template="""
    Sugiro uma cidade dado o meu interesse por {interesse}.
    {output_format}
    """,
    input_variables=["interesse"],
    partial_variables={"output_format": parser_destino.get_format_instructions()}
)

prompt_template_restaurante = PromptTemplate(
    template="""
    Sugiro restaurantes populares entre locais em {cidade}
    {output_format}
    """,
    partial_variables={"output_format": parser_restaurante.get_format_instructions()}
)

prompt_template_cultural = PromptTemplate(
    template="Sugira atividades e locais culturais em {cidade}"
)

chain_1 = prompt_template_destino | llm | parser_destino
chain_2 = prompt_template_restaurante | llm | parser_restaurante
chain_3 = prompt_template_cultural | llm | StrOutputParser()

main_chain = (chain_1 | chain_2 | chain_3)


response = main_chain.invoke(
    {'interesse': "praias"}
)

print(response)
