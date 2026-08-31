import os
import asyncio
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc


BASE_DIR = Path(__file__).resolve().parent.parent
WORKING_DIR = BASE_DIR / "rag_storage_test"


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


async def main():

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured.")

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

    await rag.initialize_storages()

    print("LightRAG storage loaded.")
    print("Running query...\n")

    question = (
        "What are the main components of Retrieval-Augmented Generation "
        "and how do they work together?"
    )

    result = await rag.aquery(
        question,
        param=QueryParam(
            mode="hybrid",
            response_type="Multiple Paragraphs",
        ),
    )

    print("QUESTION:")
    print(question)

    print("\nANSWER:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())