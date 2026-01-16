# core/engine.py
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages.tool import ToolMessage

from core.loaders import load_llm, load_collection, load_embedder
from core.router import route_semantic
from core.rag import (
    retrieve,
    simple_rerank,
    build_context_guarded,
    is_compute_query
)
from core.reflection import reflection_rewrite
from core.tools import GENERAL_TOOLS

import base64

# ===============================
# STREAMING (giữ nguyên logic cũ)
# ===============================
def stream_answer(llm, messages):
    try:
        acc = ""
        for chunk in llm.stream(messages):
            token = getattr(chunk, "content", None)
            if token:
                acc += token
                yield acc
        if not acc:
            resp = llm.invoke(messages)
            yield (resp.content or "")
    except Exception:
        resp = llm.invoke(messages)
        yield (resp.content or "")

# ===============================
# PROMPT BUILDERS (y nguyên app.py)
# ===============================
def make_rag_messages(query, context, lang="Vietnamese (Tiếng Việt)"):
    sys = f"""You are TungTomChat. Answer ONLY using context below.
- Do NOT use outside knowledge.
- If context is insufficient, say so.
- Answer in {lang}.
- Include at least one inline citation like [subject/section]."""
    human = f"QUESTION:\n{query}\n\nCONTEXT:\n{context}\n\nANSWER:"
    return [SystemMessage(content=sys), HumanMessage(content=human)]

def make_compute_messages(query, ctx_hint, lang="Vietnamese (Tiếng Việt)"):
    sys = f"""You are TungTomChat (Compute Agent).
- Solve technical calculation problems using standard formulas/algorithms.
- If a parameter is missing, list and STOP, do not invent.
- Present: (1) Input data, (2) Formula/Algorithm, (3) Step-by-step calculation, (4) Result.
- Answer in {lang}."""
    human = f"Question:\n{query}\n\nGIVEN (if any):\n{(ctx_hint or '(empty)')[:1500]}\n\nINSTRUCTION: as above."
    return [SystemMessage(content=sys), HumanMessage(content=human)]

# ===============================
# CHITCHAT (tool calling)
# ===============================
def answer_chitchat_with_tools(question, debug=False):
    llm = load_llm()
    llm_tools = llm.bind_tools(GENERAL_TOOLS)

    sys = "You are TungTomChat, a friendly assistant. Answer concisely in Vietnamese."
    human = f"Question:\n{question}\n\nAnswer:"
    msgs = [SystemMessage(content=sys), HumanMessage(content=human)]

    first = llm_tools.invoke(msgs)
    logs = []

    if getattr(first, "tool_calls", None):
        msgs.append(first)
        for tc in first.tool_calls:
            tool_name, tool_args = tc["name"], tc["args"]
            tool_obj = next((t for t in GENERAL_TOOLS if t.name == tool_name), None)
            if tool_obj:
                out = tool_obj.invoke(tool_args)
                msgs.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
                if debug:
                    logs.append(f"{tool_name} -> {out}")
        return msgs, logs
    else:
        return msgs, logs

# ===============================
# IMAGE CHAT (y nguyên app.py)
# ===============================
def chat_engine_image(message, image_base64, image_ext):
    llm = load_llm()

    system_message = SystemMessage(
        content=(
            "You are an AI learning assistant. "
            "Analyze the image carefully and solve the problem step by step. "
            "When you solve the problem, you also need to calculate and show your work. "
            "Always provide the final answer of the problem following to the problem's requirement. "
        )
    )

    human_message = HumanMessage(
        content=[
            {"type": "text", "text": message},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_ext};base64,{image_base64}"
                }
            }
        ]
    )

    answer = ""
    for partial in stream_answer(llm, [system_message, human_message]):
        answer = partial

    return {
        "answer": answer,
        "route": "image"
    }

# ===============================
# MAIN CHAT ENGINE (TEXT)
# ===============================
def chat_engine(message, history, debug=False):
    llm = load_llm()
    collection = load_collection()
    embedder = load_embedder()

    #reflection
    rewritten = reflection_rewrite(history, message)

    #routing
    route, conf = route_semantic(rewritten)

    #CHITCHAT
    if route.lower() == "chitchat":
        msgs, logs = answer_chitchat_with_tools(rewritten, debug)
        answer = ""
        for partial in stream_answer(llm, msgs):
            answer = partial
        return {
            "answer": answer,
            "route": route,
            "confidence": conf,
            "logs": logs if debug else None
        }

    #COMPUTE
    if route.lower() in ["hedieuhanh", "xulyanh"] and is_compute_query(rewritten, route):
        cands = retrieve(collection, embedder, rewritten, subject_hint=route)
        ranked = simple_rerank(rewritten, cands)
        ctx = build_context_guarded(ranked)

        answer = ""
        for partial in stream_answer(llm, make_compute_messages(rewritten, ctx)):
            answer = partial

        return {
            "answer": answer,
            "route": route,
            "confidence": conf
        }

    #RAG
    subject = None if route.lower() == "unknown" else route
    cands = retrieve(collection, embedder, rewritten, subject_hint=subject)
    ranked = simple_rerank(rewritten, cands)
    ctx = build_context_guarded(ranked)

    answer = ""
    for partial in stream_answer(llm, make_rag_messages(rewritten, ctx)):
        answer = partial

    return {
        "answer": answer,
        "route": route,
        "confidence": conf
    }
