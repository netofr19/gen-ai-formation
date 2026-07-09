import os
import httpx
import pip_system_certs.wrapt_requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from datetime import datetime
import time

load_dotenv()
print(f"✅ SSL: {os.getenv('SSL_CERT_FILE')}")

def logs(status, info):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - [{status}] - [{info}]")

def carregar_documentos(path="./data"):
    """
    Carrega documentos PDF do diretório definido.
    Args:
        path (str): caminho do diretório onde se encontram os arquivos pdf.
    Returns:
        list: Lista de documentos carregados. Retorna lista vazia em caso de erro.
    """
    try:
        loader = PyPDFDirectoryLoader(path)
        documentos = loader.load()
        logs("INFO", f"{len(documentos)} documentos carregados")
    except Exception as e:
        documentos = []
        logs("ERROR", f"{e}")
    return documentos

def dividir_documentos(docs):
    """
    Divide documentos em chunks menores para processamento.

    Args:
        docs (list): Lista de documentos a serem divididos.

    Returns:
        list: Lista de chunks gerados. Retorna lista vazia em caso de erro.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = text_splitter.split_documents(docs)
        logs("INFO", f"{len(chunks)} chunks gerados")
    except Exception as e:
        chunks = []
        logs("ERROR", f"{e}")

    return chunks

def config_embeddings(model="openai/text-embedding-3-small"):
    """
    Configura o modelo de embeddings da OpenAI via OpenRouter.

    Args:
        model (str): Nome do modelo de embeddings. Default: "openai/text-embedding-3-small".

    Returns:
        OpenAIEmbeddings | None: Instância configurada do modelo, ou None em caso de erro.
    """
    try:
        ssl_cert = os.getenv("SSL_CERT_FILE")
        http_client = httpx.Client(verify=ssl_cert) if ssl_cert else None
        embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
            model=model,
            http_client=http_client
        )
        logs("INFO", f"Modelo de embeddings configurado = {model}")
    except Exception as e:
        embeddings = None
        logs("ERROR", f"{e}")
    return embeddings

def criar_vector_store(chunks, embeddings, directory="./chroma_db"):
    """
    Cria um vector store Chroma a partir de chunks e modelo de embeddings.

    Args:
        chunks (list): Lista de chunks de documentos.
        embeddings (OpenAIEmbeddings): Modelo de embeddings configurado.
        directory (str): Diretório para persistência do vector store. Default: "./chroma_db".

    Returns:
        Chroma | None: Instância do vector store criado, ou None em caso de erro.
    """
    try:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        logs("INFO", f"Vectorstore criado no diretório: {directory}")
    except Exception as e:
        vector_store = None
        logs("ERROR", f"{e}")

if __name__ == "__main__":
    inicio = time.time()

    documentos = carregar_documentos()
    chunks = dividir_documentos(docs=documentos)
    embed_model = config_embeddings()
    criar_vector_store(chunks, embed_model)

    fim = time.time()
    tempo_total = fim - inicio

    print(f"Tempo total de execução: {tempo_total:.2f} segundos")