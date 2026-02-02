

import os
import json
import requests
import streamlit as st
from openai import OpenAI  

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Conversational Fact Checker", layout="centered")
def render_bubble(text: str, align: str):
    if align == "right":
        left, right = st.columns([1, 3])
        with right:
            st.markdown(
                f"""
                <div style="
                    background:#2a1d12;
                    color:#ffe6cc;
                    padding:12px 16px;
                    border-radius:14px;
                    border:1px solid #ffb36b;
                    text-align:right;
                    margin-left:auto;
                    max-width:95%;
                ">{text}</div>
                """,
                unsafe_allow_html=True
            )
    else:
        left, right = st.columns([3, 1])
        with left:
            st.markdown(
                f"""
                <div style="
                    background:#0f1c2e;
                    color:#e6f0ff;
                    padding:12px 16px;
                    border-radius:14px;
                    border:1px solid #3b82f6;
                    text-align:left;
                    max-width:95%;
                ">{text}</div>
                """,
                unsafe_allow_html=True
            )



st.title("Data4Good Fact Checker")
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:12px;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/0/04/ChatGPT_logo.svg" width="36">
        <body style="margin:0;">Current Model: GPT-5 Mini</body>
        <body style="margin:2;">\n</body>
    </div>
    """,
    unsafe_allow_html=True
)


# ----------------------------
# Utilities
# ----------------------------
def is_question(text: str) -> bool:
    text = text.strip().lower()
    if "?" in text:
        return True
    starters = [
        "what", "why", "how", "when", "where", "who",
        "is ", "are ", "can ", "does ", "do ", "did ",
        "should ", "could ", "would "
    ]
    return any(text.startswith(s) for s in starters)

def collect_context(chat_history):
    chunks = []
    for msg in reversed(chat_history[:-1]):
        if msg["role"] == "user" and msg["is_question"]:
            break
        if msg["role"] == "user":
            chunks.append(msg["text"])
    return "\n".join(reversed(chunks))

def build_classifier_input(answer, question, context):
    return (
        f"Answer: {answer.strip()}\n"
        f"Question: {question.strip()}\n"
        f"Context: {context.strip()}"
    )

# Fixed the variable declarations (removed trailing commas)
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION")
AZURE_DEPLOYMENT_NAME = os.environ.get("AZURE_DEPLOYMENT_NAME")

# Create the Azure OpenAI client once and reuse it
client = OpenAI(
    base_url=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
)

# ----------------------------
# Azure OpenAI GPT-5 Mini Model
# ----------------------------
def answer_question_hf(question: str, context: str) -> str:
    if context.strip() == "":
        prompt = f"""
                QUESTION:
                {question}

                One Line ANSWER:
                """.strip()
    else:
        prompt = f"""
                One Line Answer the question.

                CONTEXT:
                {context}

                QUESTION:
                {question}

                One Line ANSWER and if you think the context is not useful, just use your own knowledge,
                                                """.strip()

    completion = client.chat.completions.create(
        model=AZURE_DEPLOYMENT_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_completion_tokens=16384,
    )

    return completion.choices[0].message.content.strip()

# ----------------------------
# Azure QCA classifier
# ----------------------------
AZURE_SCORING_URI = os.environ.get("AZURE_ML_SCORING_URL")
print("Scoring URI:", AZURE_SCORING_URI)
AZURE_ML_KEY = os.environ.get("AZURE_ML_API_KEY")

def classify_answer(text_input: str):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AZURE_ML_KEY}",  # Use AZURE_ML_KEY instead of AZURE_API_KEY
    }

    payload = {
        "inputs": [text_input],
        "max_len": 512
    }

    resp = requests.post(
        AZURE_SCORING_URI,
        headers=headers,
        json=payload,
        timeout=60
    )

    if resp.status_code != 200:
        raise RuntimeError(resp.text)
    print(resp.text)
    return resp.text

# ----------------------------
# UI helpers
# ----------------------------
def render_verdict(label):
    if label == "contradiction":
        st.error(
            "⚠️ **The generated answer may be false or misleading.** "
            "Please consult additional reliable sources."
        )
    elif label == "factual":
        st.success(
            "✅ **The answer appears consistent with the provided context.**"
        )
    else:
        st.info(
            "ℹ️ **There is insufficient context to verify this answer.**", label
        )

# ----------------------------
# Session state
# ----------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# ----------------------------
# Render chat
# ----------------------------
for msg in st.session_state.chat:
    if msg["role"] == "assistant":
        with st.chat_message(
            "assistant",
            avatar="assets/chatgpt.jpg",  # temporary fallback emoji
        ):
            #st.markdown("**GPT-5 Mini**")
            #st.write(msg["text"])
            render_bubble(msg["text"], "left")

    elif msg["role"] == "user":
        with st.chat_message("user"):
            #st.write(msg["text"])
            render_bubble(msg["text"], "right")

    else:  # system messages
        with st.chat_message("assistant", avatar="🛡️"):
            if msg.get("label"):
                render_verdict(msg["label"])


# ----------------------------
# Chat input
# ----------------------------
user_input = st.chat_input("Type a message")

if user_input:
    q_flag = is_question(user_input)

    st.session_state.chat.append({
        "role": "user",
        "text": user_input,
        "is_question": q_flag,
    })

    if q_flag:
        try:
            status = st.status("🔍 Answering and verifying...", expanded=True)

            context = collect_context(st.session_state.chat)

            status.write("🧠 Generating answer (GPT-5 Mini)")
            answer = answer_question_hf(user_input, context)

            st.session_state.chat.append({
                "role": "assistant",
                "text": answer,
                "is_question": False,
            })

            status.write("🛡️ Running fact-check classifier")
            clf_input = build_classifier_input(
                answer=answer,
                question=user_input,
                context=context
            )

            clf_result = json.loads(classify_answer(clf_input))
            label = clf_result.get("pred_label", "irrelevant")

            st.session_state.chat.append({
                "role": "system",
                "text": "Fact-check result",
                "label": label,
                "is_question": False,
            })

            status.update(label="✅ Done", state="complete")


        except Exception as e:
            st.session_state.chat.append({
                "role": "system",
                "text": f"Verification failed: {e}",
                "label": None,
                "is_question": False,
            })

    st.rerun()