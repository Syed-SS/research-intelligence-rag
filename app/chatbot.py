import os
import asyncio
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent.parent
WORKING_DIR = BASE_DIR / "rag_storage_test"


# ---------------------------------------------------------
# OpenAI LLM
# ---------------------------------------------------------

async def llm_model_func(
    prompt,
    system_prompt=None,
    history_messages=None,
    **kwargs,
):
    return await openai_complete_if_cache(
        model="gpt-4o-mini",
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENAI_API_KEY"),
        **kwargs,
    )


# ---------------------------------------------------------
# Create LightRAG
# ---------------------------------------------------------

@st.cache_resource
def create_rag():

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. "
            "Please configure it in your .env file."
        )

    rag = LightRAG(
        working_dir=str(WORKING_DIR),
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model="text-embedding-3-small",
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
        ),
        llm_model_name="gpt-4o-mini",
    )

    asyncio.run(rag.initialize_storages())

    return rag


# ---------------------------------------------------------
# Query LightRAG
# ---------------------------------------------------------

def ask_rag(rag, question):

    return asyncio.run(
        rag.aquery(
            question,
            param=QueryParam(
                mode="hybrid",
                response_type="Multiple Paragraphs",
            ),
        )
    )


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

st.set_page_config(
    page_title="Research Intelligence RAG",
    page_icon="🔬",
    layout="centered",
)


st.title("🔬 Research Intelligence RAG")

st.markdown(
    """
Ask questions about the research-paper knowledge base.

**Pipeline:**  
Research Papers → LightRAG Retrieval → Retrieved Context → OpenAI LLM → Answer
"""
)

st.divider()


# ---------------------------------------------------------
# Initialize RAG
# ---------------------------------------------------------

try:
    rag = create_rag()
    st.success("LightRAG knowledge base loaded.")

except Exception as e:
    st.error(f"Unable to initialize LightRAG: {e}")
    st.stop()


# ---------------------------------------------------------
# Chat history
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------
# Chat input
# ---------------------------------------------------------

question = st.chat_input(
    "Ask a research question..."
)


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching research knowledge base..."):

            try:

                answer = ask_rag(
                    rag,
                    question,
                )

                st.markdown(answer)

                st.caption(
                    "Retrieved using LightRAG • Generated with OpenAI GPT-4o-mini"
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as e:

                st.error(
                    f"Query failed: {e}"
                )