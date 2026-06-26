import os
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import PromptTemplate


# CONFIGURAÇÕES GERAIS

# carregamento da chave como variável de ambiente
_ =  load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

PERSIST_DIRECTORY = './chroma_rh'
EMBEDDING_MODEL = 'openai/text-embedding-3-small'
LLM_MODEL = 'meta-llama/llama-3.1-8b-instruct'
API_BASE_URL = 'https://openrouter.ai/api/v1'

LOG_DIR = 'logs'
timestamp_execucao = datetime.now().strftime("%Y%m%d_%H_%M_%S")
LOG_FILE = os.path.join(LOG_DIR, f"app_{timestamp_execucao}.log")

os.makedirs(LOG_DIR, exist_ok=True)

def log(message, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] - [{level}] - [{message}]\n"

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)


# LEITURA DE DOCUMENTOS
@st.cache_data
def carregar_documentos():
    """
    Carregar PDFs de políticas interna de RH
    """

    caminhos = [
        './data/politica_ferias.pdf',
        './data/politica_home_office.pdf',
        './data/codigo_conduta.pdf'
    ]

    documentos = []

    for caminho in caminhos:
        loader = PyPDFLoader(caminho)
        docs = loader.load()

        for doc in docs:
            doc.metadata['documento'] = caminho

        documentos.extend(docs)

    return documentos


# GERAÇÃO DE CHUNKS
def gerar_chunks(documentos):
    """
    Geração de chunks a partir dos documentos carregados

    Args:
        documentos (list): lista de documentos carregados da base interna de RH.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150
    )

    return splitter.split_documents(documentos)


# ENRIQUECIMENTO COM METADADOS
def enriquecer_chunks(chunks):
    """
    Classificação dos chunks por categoria semântica

    Args:
        chunks (list): Lista com chunks gerados a partir dos documentos carregados.
    
    Returns:
        list: Lista enriquecida de chunks.
    """

    for chunk in chunks:
        texto = chunk.page_content.lower()

        if 'férias' in texto:
            chunk.metadata['categoria'] = 'ferias'
        elif 'home office' in texto or 'remoto' in texto:
            chunk.metadata['categoria'] = 'home_office'
        elif 'conduta' in texto or 'ética' in texto:
            chunk.metadata['categoria'] = 'conduta'
        else:
            chunk.metadata['categoria'] = 'geral'

    return chunks


# VECTOR STORE
@st.cache_resource
def criar_vectorstore(_chunks):
    """Criação ou carregamento do banco vetorial. O parâmetro _chunks não entra no hash do cache.

    Args:
        _chunks (list): Coleção de chunks com enriquecimento de metadados.

    Returns:
        Chroma: Instância do vectorstore persistido em disco.
    """

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=API_KEY,
        openai_api_base=API_BASE_URL
    )

    vectorstore = Chroma.from_documents(
        documents=_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    return vectorstore


# RERANKING
def rerank_documentos(pergunta, documentos, llm):
    """Reordena documentos por relevância usando LLM como juiz.

    Cada documento recebe um score de 0 a 1 baseado na relevância
    em relação à pergunta, e a lista é retornada em ordem decrescente.

    Args:
        pergunta (str): Pergunta do usuário.
        documentos (list[Document]): Documentos recuperados do vectorstore.
        llm (BaseChatModel): Modelo de linguagem para avaliar relevância.

    Returns:
        list[Document]: Documentos reordenados do mais ao menos relevante.
    """

    prompt_rerank = PromptTemplate(
        input_variables=["pergunta", "texto"],
        template="""
        Você é um especialista em políticas interna de RH.

        Pergunta do usuário:
        {pergunta}

        Trecho do documento:
        {texto}

        Avalie a relevância desse trecho para responder a pergunta.
        Responda apenas com um valor numérico entre 0 e 1. Não adicione nada mais além do valor numérico na resposta. Utilize o ponto (.) como separador de casa decimal.

        Exemplo:
        Pergunta: "Quais são as regras para home office?"
        Trecho: "A política de home office permite que os funcionários trabalhem remotamente até
        dois dias por semana."
        Resposta: 0.9

        Resposta:"""
    )

    documentos_com_score = []

    for doc in documentos:
        score = llm.invoke(
            prompt_rerank.format(
                pergunta=pergunta,
                texto=doc.page_content
            )
        ).content

        print(f"\nContent: {doc.page_content}\nScore: {score}\n")

        try:
            score = float(score)
        except:
            print("Não foi possível converter a análise de Reranking para float!")
            score = 0.0

        documentos_com_score.append((score, doc))

    # ordenação do mais relevante para o menos relevante
    documentos_ordenados = sorted(
        documentos_com_score,
        key=lambda x: x[0],
        reverse=True
    )

    # retorna apenas os documentos
    return [doc for _, doc in documentos_ordenados]


# PIPELINE COMPLETA DE RAG
def responder_pergunta(pergunta, vectorstore):
    """Responde uma pergunta usando RAG com reranking.

    Recupera documentos relevantes do vectorstore, aplica reranking via LLM
    e gera a resposta final com base nos top 4 documentos mais relevantes.

    Args:
        pergunta (str): Pergunta do usuário.
        vectorstore (Chroma): Vectorstore com os documentos indexados.

    Returns:
        tuple[str, list[Document]]: Resposta gerada e documentos utilizados como contexto.
    """

    llm_model = ChatOpenAI(
    model=LLM_MODEL,
    api_key=API_KEY,
    base_url=API_BASE_URL
    )

    # recuperação inicial (top-k mais alto)
    documentos_recuperados = vectorstore.similarity_search(
        query=pergunta,
        k=8
    )

    # reranking
    documentos_rerankeados = rerank_documentos(
        pergunta=pergunta,
        documentos=documentos_recuperados,
        llm=llm_model
    )

    # seleção dos melhores documentos
    contexto_final = documentos_rerankeados[:4]

    # prompt final
    contexto_texto = "\n\n".join(
        [doc.page_content for doc in contexto_final]
    )

    prompt_final = f"""
    Você é um agente de RH corporativo.
    Responda APENAS com base nas políticas internas abaixo.
    
    Contexto:
    {contexto_texto}

    Pergunta:
    {pergunta}
    """

    resposta = llm_model.invoke(prompt_final)

    return resposta.content, contexto_final

# INTERFACE STREAMLIT

st.set_page_config(page_title="CHAT de RH com RAG", layout="wide")
st.title("Chat de RH - Políticas Internas")
log("Aplicação iniciada")

pergunta = st.text_input("Digite sua pergunta sobre políticas internas de RH: ")

if pergunta:
    log(f"Pergunta enviada pelo usuário: {pergunta}")
    with st.spinner("Consultando políticas internas..."):
        log("Carregamento de documentos")
        documentos = carregar_documentos()
        log("Geração de Chunks")
        chunks = gerar_chunks(documentos)
        log("Enriquecimento de chunks")
        chunks = enriquecer_chunks(chunks)
        log("Criação do vectorstore")
        vectorstore = criar_vectorstore(chunks)

        log("Geração da resposta")
        resposta, fontes = responder_pergunta(pergunta, vectorstore)
        log(f"Resposta gerada: {resposta}")

        log("Encerramento da conversa")

    st.subheader("Resposta")
    st.write(resposta)

    st.subheader("Fontes Utilizadas")
    for i, doc in enumerate(fontes, start=1):
        st.markdown(f"===== TRECHO {i} =====")
        st.write(f"Documentos: {doc.metadata.get('documento')}")
        st.write(f"Categoria: {doc.metadata.get('categoria')}")
        st.write(doc.page_content)
        st.divider()