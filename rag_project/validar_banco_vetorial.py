import pip_system_certs.wrapt_requests
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
    model="openai/text-embedding-3-small"
)

db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

total = db._collection.count()
print(f"Total de documentos armazenados: {total}")

amostra = db._collection.get(limit=3)
for i, doc in enumerate(amostra["documents"]):
    print(f"\n--- Chunk {i+1} ---")
    print(doc[:300])
