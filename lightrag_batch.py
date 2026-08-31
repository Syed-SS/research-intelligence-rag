import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "data" / "processed"
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

    files = [
        INPUT_DIR / f"RAG-{i:03d}.txt"
        for i in range(2, 11)
    ]

    print("\n=== LightRAG Batch: RAG-002 ? RAG-010 ===\n")

    for input_file in files:
        if not input_file.exists():
            raise FileNotFoundError(f"Missing file: {input_file}")

        doc_id = input_file.stem

        print(f"\n{'=' * 60}")
        print(f"Processing: {doc_id}")
        print(f"Characters: {input_file.stat().st_size}")
        print(f"{'=' * 60}")

        text = input_file.read_text(encoding="utf-8")

        try:
            result = await rag.ainsert(
                text,
                ids=doc_id,
                file_paths=str(input_file),
            )
            print(f"Completed: {doc_id}")
            print(f"Result: {result}")

        except Exception as e:
            print(f"\nFAILED: {doc_id}")
            print(f"Error: {type(e).__name__}: {e}")
            raise

    print("\n" + "=" * 60)
    print("ALL RAG-002 ? RAG-010 DOCUMENTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
