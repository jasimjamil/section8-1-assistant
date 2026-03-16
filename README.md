# Section 8.1 Legal Assistant

Reinforcement Fine-Tuned AI Model for **Section 8.1 of the Income Tax Assessment Act 1997 (General Deductions)**.

## How it works

- **DPO (Reinforcement Fine-Tuning):** Trained the model to prefer correct answers and reject wrong ones
- **SFT (Content Memorization):** Memorized the exact text of Section 8.1 into the model
- **Base Model:** Qwen2.5-7B-Instruct
- **HuggingFace Model:** [muhammadjasim12/rainforcejasim](https://huggingface.co/muhammadjasim12/rainforcejasim)

## Features

- Answers questions about Section 8.1 accurately
- Refuses questions about other sections (8.2, 8.3, Division 7A, etc.)
- Corrects wrong details in questions
- Does not hallucinate or add information not in Section 8.1

## Setup

1. Clone this repo
2. Add your HuggingFace token in Streamlit secrets
3. Deploy on Streamlit Cloud

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add secret: `HF_TOKEN = "hf_your_token"`
5. Deploy
