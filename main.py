from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import certifi
import os

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
os.environ['SSL_CERT_FILE'] = certifi.where()

llm = ChatOpenAI(
    model="qwen/qwen3-next-80b-a3b-instruct:free",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

print(llm.invoke("Explique o que é o OpenRouter.").content)
