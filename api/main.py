# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from core.engine import chat_engine, chat_engine_image

app = FastAPI(
    title="TungTomChat Backend",
    description="FastAPI backend for TungTomChat (RAG + Tools + Image Reasoning)",
    version="1.0.0"
)

# ===============================
# REQUEST / RESPONSE SCHEMAS
# ===============================
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    debug: Optional[bool] = False

class ChatResponse(BaseModel):
    answer: str
    route: str
    confidence: float
    logs: Optional[List[str]] = None


class ImageChatRequest(BaseModel):
    message: str
    image_base64: str
    image_ext: str   # png | jpg | jpeg
    debug: Optional[bool] = False

class ImageChatResponse(BaseModel):
    answer: str
    route: str


# ===============================
# TEXT CHAT ENDPOINT
# ===============================
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Main text-based chatbot endpoint.
    """
    result = chat_engine(
        message=req.message,
        history=[m.model_dump() for m in req.history],
        debug=req.debug
    )

    return result


# ===============================
# IMAGE CHAT ENDPOINT
# ===============================
@app.post("/chat_image", response_model=ImageChatResponse)
def chat_image(req: ImageChatRequest):
    """
    Image-based chatbot endpoint.
    """
    result = chat_engine_image(
        message=req.message,
        image_base64=req.image_base64,
        image_ext=req.image_ext
    )
    return result


# ===============================
# HEALTH CHECK
# ===============================
@app.get("/health")
def health():
    return {"status": "ok"}
