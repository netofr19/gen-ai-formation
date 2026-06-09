from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
import asyncio
import os

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

class Rota(TypedDict):
    destino: Literal["praia", "montanha"]

class Estado(TypedDict):
    query: str
    destino: Rota
    resposta: str

async def no_roteador(estado: Estado, config=RunnableConfig):
    return {"destino": await roteador.ainvoke({"query": estado["query"]}, config)}

async def no_praia(estado: Estado, config=RunnableConfig):
    return {"resposta": await cadeia_praia.ainvoke({"query": estado["query"]}, config)}

async def no_montanha(estado: Estado, config=RunnableConfig):
    return {"resposta": await cadeia_montanha.ainvoke({"query": estado["query"]}, config)}

def escolher_no(estado: Estado)-> Literal["praia", "montanha"]:
    print(estado["destino"])
    return "praia" if estado["destino"]["destino"] == "praia" else "montanha"

# Criando a instância do modelo de linguagem
llm = ChatOpenAI(
    model="meta-llama/llama-3.1-8b-instruct",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

prompt_consultor_praia = ChatPromptTemplate.from_messages(
    [
        ("system", "Sempre se apresente como Sr. Praia no início da conversa. Você é um especialista e m viagens com destinos para praias"),
        ("human", "{query}")
    ]
)

prompt_consultor_montanha = ChatPromptTemplate.from_messages(
    [
        ("system", "Sempre se apresente como Sr. Montanha no início da conversa. Você é um especialista e m viagens com destinos para montanhas e atividades radicias"),
        ("human", "{query}")
    ]
)

cadeia_praia = prompt_consultor_praia | llm | StrOutputParser()
cadeia_montanha = prompt_consultor_montanha | llm | StrOutputParser()

prompt_roteador = ChatPromptTemplate.from_messages(
    [
        ("system", "Respnda apenas com 'praia' ou 'montanha'"),
        ('human', "{query}")
    ]
)

roteador = prompt_roteador | llm.with_structured_output(Rota)

grafo = StateGraph(Estado)
grafo.add_node("rotear", no_roteador)
grafo.add_node("praia", no_praia)
grafo.add_node("montanha", no_montanha)

grafo.add_edge(START, "rotear")
grafo.add_conditional_edges("rotear", escolher_no)
grafo.add_edge("praia", END)
grafo.add_edge("montanha", END)

app = grafo.compile()

async def main():
    resposta = await app.ainvoke(
        {"query": "Quero ir para praias bonitas"}
    )

    print(resposta["resposta"])

asyncio.run(main())