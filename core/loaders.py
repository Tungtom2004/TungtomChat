# core/loaders.py
import os
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings

from langchain_openai import AzureChatOpenAI
from embeddings import get_embedding

load_dotenv()

# ===============================
# ENV
# ===============================
DB_PATH = os.getenv("DB_PATH", ".chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ptit_giaotrinh_2")

# ===============================
# SINGLETON OBJECTS
# ===============================
_llm = None
_embedder = None
_collection = None
_reranker = None

# ===============================
# LOADERS
# ===============================
def load_llm():
    """
    Load Azure OpenAI LLM (singleton)
    """
    global _llm
    if _llm is None:
        _llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_VERSION"),
        )
    return _llm


def load_embedder():
    """
    Load embedding model (singleton)
    """
    global _embedder
    if _embedder is None:
        _embedder = get_embedding()
    return _embedder


def load_collection():
    """
    Load ChromaDB collection (singleton)
    """
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(
            path=DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        _collection = client.get_collection(name=COLLECTION_NAME)
    return _collection


def load_reranker():
    """
    Optional reranker (nếu có rerank.py)
    """
    global _reranker
    if _reranker is None:
        try:
            from rerank import Reranker
            _reranker = Reranker()
        except Exception:
            _reranker = None
    return _reranker
