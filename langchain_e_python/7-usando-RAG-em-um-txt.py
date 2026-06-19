from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Carregamento da chave de API e configuração do modelo de LLM
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
print("Chave carregada:", api_key[:5] + "...")

llm = ChatOpenAI(
    model="meta-llama/llama-3.1-8b-instruct",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

# criação dos embeddings
embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1"
)
documento = TextLoader(
    "docs/GTB_gold_Nov23.txt",
    encoding="utf-8"
).load()

pedacos = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
).split_documents(documento)

dados_recuperados = FAISS.from_documents(
    pedacos,
    embeddings
).as_retriever(search_kwargs={"k": 2})

prompt_consulta_seguro = ChatPromptTemplate.from_messages(
    [
        ("system", "Responsa usando exclusivamente o conteúdo fornecido como contexto"),
        ("human", "{query}\n\nContexto:\n{context}\n\nResposta:")
    ]
)

cadeia = prompt_consulta_seguro | llm | StrOutputParser()

def responder(pergunta: str):
    trechos = dados_recuperados.invoke(pergunta)
    contexto = "\n\n".join(trecho.page_content for trecho in trechos)
    return cadeia.invoke(
        {
            "query": pergunta,
            "context": contexto
        }
    ), contexto

pergunta = "O que fazer caso eu seja roubado?"
resposta, contexto = responder(pergunta)

print(f"""
PERGUNTA: {pergunta}

CONTEXTO: {contexto}

RESPOSTA: {resposta}
""")