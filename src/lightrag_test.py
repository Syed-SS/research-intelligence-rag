import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "RAG-001.txt"
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

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    WORKING_DIR.mkdir(parents=True, exist_ok=True)

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

    text = INPUT_FILE.read_text(encoding="utf-8")

    print(f"Loaded: {INPUT_FILE.name}")
    print(f"Characters: {len(text)}")
    print("Inserting document into LightRAG...")

    result = await rag.ainsert(
        text,
        ids="RAG-001",
        file_paths=str(INPUT_FILE),
    )

    print("LightRAG insertion completed.")
    print(f"Result: {result}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
