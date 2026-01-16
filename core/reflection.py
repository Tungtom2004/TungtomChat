# core/reflection.py
from langchain_core.messages import SystemMessage, HumanMessage
from core.loaders import load_llm

def build_short_history(messages, k_turns=6, max_chars=1000):
    """
    Rút gọn lịch sử chat để đưa vào prompt reflection
    """
    hist = []
    for m in messages[-2 * k_turns:]:
        role = m.get("role", "")
        content = m.get("content", "")
        hist.append(f"{role}:\n{content}")
    text = "\n".join(hist)
    return text[-max_chars:] if text else "(blank)"

def reflection_rewrite(history: list, last_user_message: str) -> str:
    """
    Rewrite câu hỏi cuối thành standalone question (Tiếng Việt)
    """
    llm = load_llm()
    short_hist = build_short_history(history)

    system_prompt = (
        "You are a rewriting assistant for questions.\n"
        "- Using the chat history and the latest user message, rewrite the LAST USER MESSAGE into a standalone Vietnamese question.\n"
        "- If the last message contains MULTIPLE QUESTIONS, KEEP ALL of them.\n"
        "- Do NOT answer the questions.\n"
        "- Output ONLY the rewritten message (no explanation).\n"
        "- Max ~400 characters.\n"
        "- If the last message is not a question, return it unchanged."
    )

    human_prompt = (
        f"CHAT HISTORY (short):\n{short_hist}\n\n"
        f"LATEST USER MESSAGE:\n{last_user_message}\n\n"
        f"STANDALONE QUESTION:"
    )

    try:
        resp = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        text = (resp.content or "").strip()
        return text if text else last_user_message
    except Exception:
        return last_user_message
