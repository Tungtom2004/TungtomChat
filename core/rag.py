# core/rag.py
import re
import numpy as np

from core.loaders import (
    load_collection,
    load_embedder,
    load_reranker
)

# ===============================
# CONFIG
# ===============================
FETCH_K = int(__import__("os").getenv("FETCH_K", "30"))
TOP_K = int(__import__("os").getenv("TOP_K", "10"))
DIST_THRES = float(__import__("os").getenv("DIST_THRES", "1.60"))

# ===============================
# RETRIEVE
# ===============================
def retrieve(collection, embedder, query, subject_hint=None, k=FETCH_K):
    qvec = embedder.embed_query(query)
    where = {"subject": subject_hint} if subject_hint else {}

    res = collection.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["metadatas", "documents", "distances"],
        **({"where": where} if where else {})
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    return list(zip(docs, metas, dists))

# ===============================
# RERANK
# ===============================
def simple_rerank(query, candidates, top_k=TOP_K):
    reranker = load_reranker()

    # === dùng reranker nếu có ===
    if reranker and candidates:
        passages = [(d or "")[:1200] for (d, _, _) in candidates]
        try:
            _, ranked_pass = reranker(query, passages)
            used = [False] * len(candidates)
            out = []
            for p in ranked_pass:
                for i, (d, m, dist) in enumerate(candidates):
                    if used[i]:
                        continue
                    if (d or "")[:1200] == p:
                        out.append((d, m, dist))
                        used[i] = True
                        break
            return out[:min(top_k, len(out))]
        except Exception:
            pass

    # === fallback heuristic (y nguyên app.py) ===
    qtok = set((query or "").lower().split())
    scored = []

    for (d, m, dist) in candidates:
        toks = set((d or "").lower().split())
        overlap = len(qtok & toks)

        dterm = 0.0
        try:
            dterm = 1 / (1 + float(dist))
        except Exception:
            pass

        score = 0.7 * overlap + 0.3 * dterm
        scored.append((score, (d, m, dist)))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in scored[:min(top_k, len(scored))]]

# ===============================
# BUILD CONTEXT (GUARDED)
# ===============================
def build_context_guarded(ranked, max_chars=900):
    if not ranked:
        return ""

    good = []
    for d, m, dist in ranked:
        try:
            if dist is None or (isinstance(dist, (int, float)) and dist <= DIST_THRES):
                good.append((d, m))
        except Exception:
            continue

    if not good:
        return ""

    blocks = []
    for d, m in good:
        head = f"[{m.get('subject', '')}/{m.get('section', '')}]"
        blocks.append(f"{head}\n{(d or '')[:max_chars]}")

    return "\n\n".join(blocks)

# ===============================
# COMPUTE QUERY DETECTOR
# ===============================
def is_compute_query(query, subject):
    ql = (query or "").lower()

    if subject and subject.lower() == "hedieuhanh":
        kws = [
            "tính", "waiting time", "turnaround", "response time",
            "throughput", "gantt", "fcfs", "sjf", "srtf",
            "rr", "round robin", "quantum", "burst", "arrival"
        ]
        return any(k in ql for k in kws) and bool(re.search(r"\d", ql))

    if subject and subject.lower() == "xulyanh":
        kws = [
            "tính", "tích chập", "convolution", "kernel",
            "3x3", "5x5", "padding", "stride",
            "sobel", "prewitt", "laplacian",
            "gradient", "magnitude", "otsu", "threshold"
        ]
        looks_num = bool(re.search(r"\d", ql)) or ("[" in ql and "]" in ql)
        return any(k in ql for k in kws) and looks_num

    return False
