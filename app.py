import streamlit as st
import requests

st.set_page_config(
    page_title="Section 8.1 Legal Assistant",
    page_icon="⚖️",
)

st.title("⚖️ Section 8.1 Legal Assistant (Free Version)")
st.markdown("---")

# ✅ FREE MODEL (WORKS ON HF)
API_URL = "https://router.huggingface.co/hf-inference/models/google/gemma-2-2b-it"

HF_TOKEN = st.secrets.get("HF_TOKEN", "")

SYSTEM_PROMPT = """You ONLY answer questions about Section 8.1 of the Income Tax Assessment Act 1997 (General Deductions). If a question is about any other section, topic, refuse politely."""


def ask(question):
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
    }

    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        "max_tokens": 200
    }

    try:
        res = requests.post(API_URL, headers=headers, json=payload)

        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]

        elif res.status_code == 503:
            return "⏳ Model loading... try again"

        else:
            return f"❌ Error {res.status_code}: {res.text}"

    except Exception as e:
        return f"🚨 {str(e)}"


# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask about Section 8.1...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask(user_input)
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
