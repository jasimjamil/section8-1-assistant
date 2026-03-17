import streamlit as st
import requests

st.set_page_config(
    page_title="Section 8.1 Legal Assistant",
    page_icon="⚖️",
    layout="centered",
)

st.title("⚖️ Section 8.1 Legal Assistant")
st.markdown("**Reinforcement Fine-Tuned Model | ITAA 1997 - Section 8.1 (General Deductions)**")
st.markdown("---")

API_URL = "https://router.huggingface.co/models/muhammadjasim12/rainforcejasim-merged"


SYSTEM_PROMPT = """You ONLY answer questions about Section 8.1 of the Income Tax Assessment Act 1997 (General Deductions). If a question is about any other section, topic, or contains wrong details about Section 8.1, refuse or correct it. Never add information not in Section 8.1."""

HF_TOKEN = st.secrets.get("HF_TOKEN", "")


def ask(question):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "do_sample": False,
            "return_full_text": False,
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            text = result[0].get("generated_text", "")
            # Clean up end tokens
            text = text.replace("<|im_end|>", "").strip()
            return text
        return "No response from model."
    elif response.status_code == 503:
        return "Model is loading... Please wait 30 seconds and try again."
    else:
        return f"Error: {response.status_code} - {response.text}"


# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question about Section 8.1...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask(user_input)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# Sidebar
with st.sidebar:
    st.header("⚖️ About")
    st.markdown("""
    This model is **reinforcement fine-tuned (DPO + SFT)** 
    exclusively on **Section 8.1** of the Income Tax 
    Assessment Act 1997 (General Deductions).
    """)

    st.markdown("---")
    st.header("✅ It will")
    st.markdown("""
    - Answer questions about Section 8.1
    - Refuse questions about other sections
    - Correct wrong details in questions
    """)

    st.markdown("---")
    st.header("❌ It will NOT")
    st.markdown("""
    - Answer questions outside Section 8.1
    - Add information not in the section
    - Make up dollar amounts or rules
    """)

    st.markdown("---")
    st.header("💡 Example Questions")
    examples = [
        "What is Section 8.1 about?",
        "What does Section 8.1(1)(a) say?",
        "Can I deduct a capital expense?",
        "What does Section 8.2 say?",
        "Does Section 8.1 have four subsections?",
        "Section 8.1(1)(b) says 'reasonably incurred' right?",
        "What is Division 7A?",
        "What is the weather today?",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state.messages.append({"role": "user", "content": ex})
            st.rerun()

    st.markdown("---")
    st.markdown("**Model:** `muhammadjasim12/rainforcejasim`")
    st.markdown("**Base:** Qwen2.5-7B-Instruct")
    st.markdown("**Training:** DPO + SFT")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
