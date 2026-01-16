import json
import os
from dotenv import load_dotenv
from llama_index.core import PromptTemplate
from embeddings import get_embedding 
from rerank import Reranker


# ============================================================
# 1) DISABLE DEFAULT LLM + SET EMBEDDING MODEL (MATCH CHROMA)
# ============================================================
from llama_index.core import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Tắt hoàn toàn OpenAI LLM mặc định
Settings.llm = None

# Dùng đúng embedding đã add vào Chroma (768-dim)
Settings.embed_model = HuggingFaceEmbeddings(
    model_name="Alibaba-NLP/gte-multilingual-base",
    model_kwargs={'device': 'cpu', 'trust_remote_code': True},
    encode_kwargs={'normalize_embeddings': True},
)


# ============================================================
# 2) IMPORT LlamaIndex + Chroma
# ============================================================
import chromadb
from chromadb.config import Settings as ChromaSettings

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

# ============================================================
# 3) Azure OpenAI — dùng để ĐÁNH GIÁ (KHÔNG dùng query)
# ============================================================
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.evaluation import (
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator
)

reranker = Reranker(model_name="Alibaba-NLP/gte-multilingual-reranker-base")



def load_llm():
    load_dotenv()
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_VERSION"),
    )


llm = load_llm()

correctness_eval = CorrectnessEvaluator(llm=llm)
faith_eval       = FaithfulnessEvaluator(llm=llm)
rel_eval         = RelevancyEvaluator(llm=llm)


# ============================================================
# 4) Load ChromaDB
# ============================================================
CHROMA_PATH = "D:/ChatBotSlidePTIT/.chroma_db"
COLLECTION_NAME = "ptit_giaotrinh_2"

client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=ChromaSettings(anonymized_telemetry=False)
)

print("\n📌 ChromaDB path:", os.path.abspath(CHROMA_PATH))
print("📌 Available Collections:", client.list_collections())

collection = client.get_collection(COLLECTION_NAME)
print("✔ Loaded collection:", COLLECTION_NAME)



vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
)

qa_template = PromptTemplate(
    """
You are an expert assistant.

Use ONLY the information provided in the context.
DO NOT use outside knowledge.
DO NOT hallucinate.

---------------------
Context:
{context_str}
---------------------

Question:
{query_str}

Answer:
"""
)

query_engine = index.as_query_engine(
    similarity_top_k=5,
    llm=llm,
    text_qa_template=qa_template
)

with open("tv.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

results = []


for item in dataset:
    question = item["question"]
    ref_ans  = item["answer"]

    print("\n==============================")
    print("❓ Question:", question)

    # ======================================================
    # 1) Retrieve bằng query engine
    # ======================================================
    try:
        response = query_engine.query(question)
    except Exception as e:
        print("⚠ Query error:", e)
        continue

    # ======================================================
    # 2) Lấy câu trả lời
    # ======================================================
    rag_answer = ""
    try:
        rag_answer = response.response  
    except:
        rag_answer = str(response) if response else ""

    # ======================================================
    # 3) Lấy context gốc từ source_nodes
    # ======================================================
    try:
        raw_contexts = [node.get_text() for node in response.source_nodes]
    except:
        raw_contexts = []

    if not rag_answer or rag_answer.strip() == "":
        print("⚠ Empty RAG answer → SKIP")
        continue

    if not raw_contexts:
        print("⚠ No context retrieved → SKIP")
        continue

    # ======================================================
    # ⭐ 4) RERANK context (bước QUAN TRỌNG NHẤT)
    # ======================================================
    ranked_scores, ranked_passages = reranker(
        query=question,
        passages=raw_contexts
    )

    # Giữ top-3 passage sau rerank (tùy bạn chỉnh)
    reranked_contexts = ranked_passages[:3]

    print("📌 Before Rerank:", len(raw_contexts), "contexts")
    print("📌 After Rerank:", len(reranked_contexts), "contexts")

    # ======================================================
    # 5) Evaluate correctness / relevance / faithfulness
    # ======================================================
    try:
        correctness = correctness_eval.evaluate(
            query=question,
            response=rag_answer,
            reference=ref_ans
        )
    except Exception as e:
        print("⚠ Correctness evaluator failed:", e)
        continue

    try:
        faithfulness = faith_eval.evaluate(
            response=rag_answer,
            contexts=reranked_contexts  # ⭐ dùng context đã rerank
        )
    except Exception as e:
        print("⚠ Faithfulness evaluator failed:", e)
        continue

    try:
        relevancy = rel_eval.evaluate(
            query=question,
            response=rag_answer,
            contexts=reranked_contexts
        )
    except Exception as e:
        print("⚠ Relevancy evaluator failed:", e)
        continue

    # ======================================================
    # 6) Lưu kết quả
    # ======================================================
    results.append({
        "question": question,
        "reference_answer": ref_ans,
        "rag_answer": rag_answer,
        "correctness_score": correctness.score,
        "faithfulness_score": faithfulness.score,
        "relevancy_score": relevancy.score,
        "retrieved_context_before": "\n---\n".join(raw_contexts),
        "retrieved_context_after_rerank": "\n---\n".join(reranked_contexts),
        "rerank_scores": ranked_scores[:3],
    })

    print("  ✔ Correctness:", correctness.score)
    print("  ✔ Faithfulness:", faithfulness.score)
    print("  ✔ Relevancy:", relevancy.score)


with open("rag_evaluation_results_2.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("\n🎉 DONE! Saved → rag_evaluation_results.json\n")
