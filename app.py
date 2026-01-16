# -*- coding: utf-8 -*-
import base64
import requests
import streamlit as st

# ===============================
# CONFIG
# ===============================
BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="ChatbotPTIT",
    page_icon="🦐",
    layout="wide"
)

st.markdown(
    "<style>.stChatMessage { font-size: 16px; }</style>",
    unsafe_allow_html=True
)

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.title("ChatbotPTIT")
    st.caption("Frontend (Streamlit)")
    st.caption("Backend: FastAPI")

# ===============================
# SESSION STATE
# ===============================
if "dialog" not in st.session_state:
    st.session_state.dialog = []

# ===============================
# HEADER
# ===============================
st.header("ChatbotPTIT")

# ===============================
# RENDER CHAT HISTORY
# ===============================
for m in st.session_state.dialog:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ===============================
# IMAGE UPLOAD
# ===============================
uploaded_file = st.file_uploader(
    "Tải ảnh lên",
    type=["png", "jpg", "jpeg", "bmp", "gif"]
)

if uploaded_file:
    st.image(
        uploaded_file.getvalue(),
        caption="Ảnh vừa tải lên",
        width=200
    )

# ===============================
# CHAT INPUT
# ===============================
user_msg = st.chat_input("Gõ câu hỏi của bạn…")

if user_msg:
    # --- show user message ---
    st.session_state.dialog.append(
        {"role": "user", "content": user_msg}
    )

    with st.chat_message("user"):
        st.markdown(user_msg)
        if uploaded_file:
            st.image(uploaded_file.getvalue(), width=200)

    # --- assistant placeholder ---
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("*(Đang suy nghĩ…)*")

        try:
            # ===============================
            # IMAGE CHAT
            # ===============================
            if uploaded_file:
                image_bytes = uploaded_file.getvalue()
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")

                ext = uploaded_file.name.split(".")[-1].lower()
                if ext not in ["png", "jpg", "jpeg"]:
                    ext = "jpeg"

                resp = requests.post(
                    f"{BACKEND_URL}/chat_image",
                    json={
                        "message": user_msg,
                        "image_base64": image_base64,
                        "image_ext": ext,
                    },
                    timeout=120,
                )

            # ===============================
            # TEXT CHAT
            # ===============================
            else:
                resp = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={
                        "message": user_msg,
                        "history": st.session_state.dialog,
                    },
                    timeout=120,
                )

            if resp.status_code != 200:
                placeholder.markdown(
                    f"Backend error: {resp.status_code}"
                )
            else:
                data = resp.json()
                answer = data.get("answer", "(no answer)")
                placeholder.markdown(answer)

                # debug info

                st.session_state.dialog.append(
                    {"role": "assistant", "content": answer}
                )

        except requests.exceptions.RequestException as e:
            placeholder.markdown(f"Không kết nối được backend: {e}")
