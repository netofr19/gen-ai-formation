import os
import httpx
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

load_dotenv()

def logs(status, info):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - [{status}] - [{info}]")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def inicializar_componentes(persist_directory="./chroma_db", embed_model_name="openai/text-embedding-3-small", llm_model_name="google/gemma-4-31b-it:free"):
    """
    Inicializa os componentes do pipeline RAG: embeddings, vector store e LLM.

    Args:
        persist_directory (str): Diretório do vector store Chroma. Default: "./chroma_db".
        embed_model_name (str): Nome do modelo de embeddings. Default: "openai/text-embedding-3-small".
        llm_model_name (str): Nome do modelo LLM. Default: "google/gemma-4-31b-it:free".

    Returns:
        tuple: (Chroma, ChatOpenAI) com o vector store e o modelo LLM, ou (None, None) em caso de erro.
    """
    try:
        ssl_cert = os.getenv("SSL_CERT_FILE")
        http_client = httpx.Client(verify=ssl_cert) if ssl_cert else None

        # configuração do modelo de embeddings
        embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
            model=embed_model_name,
            http_client=http_client
        )
        logs("INFO", f"Modelo de Embeddings configurado: {embed_model_name}")

        # carregamento do banco de dados Chroma
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        logs("INFO", "Vector Store carregado com sucesso.")

        # Configuração da LLM via OpenRouter
        llm_model = ChatOpenAI(
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
            model=llm_model_name,
            http_client=http_client,
            temperature=0.3
        )
        logs("INFO", f"Modelo de LLM configurado: {llm_model_name}")

        return vector_store, llm_model
    
    except Exception as e:
        logs("ERROR", f"Erro na inicialização: {e}")

        return None, None
    
def criar_rag_chain(vector_store, llm):
    """
    Cria a cadeia RAG (Retrieval-Augmented Generation) para responder perguntas com base no contexto recuperado.

    Args:
        vector_store (Chroma): Vector store com os documentos indexados.
        llm (ChatOpenAI): Modelo LLM configurado.

    Returns:
        RunnableSequence | None: Cadeia RAG pronta para uso, ou None em caso de erro.
    """
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        system_prompt = (
            "Você é um assistente virtual prestativo e especialista em fármacos. Use estritamente os seguintes pedaços de contexto recuperados para responder à pergunta do usuário. Se você não souber a resposta ou se ela não estiver no contexto, diga explicitamente que não encontrou a informação na sua base de conhecimento. Não invente fatos.\n\nContexto:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        rag_chain = (
            {
                "context": retriever | format_docs,
                "input": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        return rag_chain
    
    except Exception as e:
        logs("ERROR", f"Erro na criação da cadeia de geração de resposta: {e}")
        return None
    
if __name__ == "__main__":
    # inicialização dos componentes
    vector_store, llm = inicializar_componentes()

    if vector_store and llm:

        chat_bot = criar_rag_chain(vector_store, llm)

        print("\n🤖 Chat RAG Inicializado! Digite 'sair' para encerrar.\n")

        while True:

            pergunta = input("👤 Você: ")
            if pergunta.lower() in ["sair", "exit", "quit"]:
                print("Até logo!")
                break

            if not pergunta.strip():
                continue

            try:
                # execução da busca + geração de resposta
                resposta = chat_bot.stream(pergunta)

                print(f"\n🤖 Assistente: {resposta}\n")
                print(f"[Fontes utilizadas: {[doc.metadata for doc in resposta['context']]}]\n")

            except Exception as e:
                logs("ERROR", f"Erro ao processar pergunta: {e}")