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

# ✅ NEW HF ROUTER API (FIXED)
API_URL = "https://router.huggingface.co/hf-inference/models/muhammadjasim12/rainforcejasim"

SYSTEM_PROMPT = """You ONLY answer questions about Section 8.1 of the Income Tax Assessment Act 1997 (General Deductions). If a question is about any other section, topic, or contains wrong details about Section 8.1, refuse or correct it. Never add information not in Section 8.1."""

# 🔐 Add your token in HF Spaces Secrets
HF_TOKEN = st.secrets.get("HF_TOKEN", "")


# ✅ ASK FUNCTION (ROBUST)
def ask(question):
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "do_sample": False,
            "return_full_text": False,
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)

        # ✅ SUCCESS
        if response.status_code == 200:
            result = response.json()

            if isinstance(result, list) and len(result) > 0:
                text = result[0].get("generated_text", "")
                return text.replace("<|im_end|>", "").strip()

            return "⚠️ No valid response from model."

        # ⏳ MODEL LOADING
        elif response.status_code == 503:
            return "⏳ Model is loading... please wait 20–30 seconds and try again."

        # ❌ COMMON ERRORS
        elif response.status_code == 404:
            return "❌ Model not found or not deployed on inference API."

        elif response.status_code == 401:
            return "🔐 Invalid or missing HuggingFace token."

        else:
            return f"❌ Error {response.status_code}: {response.text}"

    except Exception as e:
        return f"🚨 Request failed: {str(e)}"


# ✅ CHAT UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
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


# ✅ SIDEBAR
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
    - Answer Section 8.1 questions  
    - Refuse other sections  
    - Correct wrong inputs  
    """)

    st.markdown("---")
    st.header("❌ It will NOT")
    st.markdown("""
    - Answer outside Section 8.1  
    - Add extra information  
    - Hallucinate rules  
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
    ]

    for ex in examples:
        if st.button(ex):
            st.session_state.messages.append({"role": "user", "content": ex})
            st.rerun()

    st.markdown("---")
    st.markdown("**Model:** `muhammadjasim12/rainforcejasim`")
    st.markdown("**Base:** Qwen2.5-7B-Instruct")
    st.markdown("**Training:** DPO + SFT")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
