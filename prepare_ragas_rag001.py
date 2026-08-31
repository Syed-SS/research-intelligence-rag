import os
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
WORKING_DIR = BASE_DIR / "rag_storage_test"
OUTPUT_FILE = BASE_DIR / "ragas_rag001_data.json"


# ============================================================
# RAG-001 EVALUATION QUESTIONS
# ============================================================

QUESTIONS = [
    {
        "id": "RAG-001-Q1",
        "question": (
            "What are the three foundational components of RAG "
            "frameworks identified in the survey?"
        ),
        "reference": (
            "The three foundational components are retrieval, "
            "generation, and augmentation techniques."
        ),
    },
    {
        "id": "RAG-001-Q2",
        "question": (
            "Why does Retrieval-Augmented Generation help address "
            "limitations of Large Language Models, particularly for "
            "knowledge-intensive tasks?"
        ),
        "reference": (
            "RAG incorporates knowledge from external databases, "
            "which enhances the accuracy and credibility of generation "
            "for knowledge-intensive tasks. It also enables continuous "
            "knowledge updates and integration of domain-specific "
            "information, while combining an LLM's intrinsic knowledge "
            "with external knowledge repositories."
        ),
    },
    {
        "id": "RAG-001-Q3",
        "question": (
            "What progression of RAG paradigms does the survey examine?"
        ),
        "reference": (
            "The survey examines the progression of Naive RAG, "
            "Advanced RAG, and Modular RAG."
        ),
    },
]


# ============================================================
# LLM
# ============================================================

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


# ============================================================
# MAIN
# ============================================================

async def main():

    # --------------------------------------------------------
    # Environment check
    # --------------------------------------------------------

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    if not WORKING_DIR.exists():
        raise FileNotFoundError(
            f"LightRAG working directory not found: {WORKING_DIR}"
        )

    # --------------------------------------------------------
    # Initialize LightRAG
    # --------------------------------------------------------

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

    print()
    print("=" * 72)
    print("RAGAS DATA PREPARATION")
    print("=" * 72)
    print("LightRAG corpus : RAG-001 ? RAG-010")
    print("Evaluation set  : RAG-001")
    print("Questions       :", len(QUESTIONS))
    print("RAGAS version   : 0.4.3")
    print("=" * 72)

    results = []

    # --------------------------------------------------------
    # Process each evaluation question
    # --------------------------------------------------------

    for item in QUESTIONS:

        question_id = item["id"]
        question = item["question"]

        print()
        print("-" * 72)
        print(question_id)
        print("-" * 72)
        print("QUESTION:")
        print(question)

        # ====================================================
        # STEP 1 — RETRIEVE CONTEXT
        # ====================================================

        print()
        print("[1/2] Retrieving context...")

        context_result = await rag.aquery_llm(
            question,
            param=QueryParam(
                mode="hybrid",
                response_type="Multiple Paragraphs",

                # Important:
                # No reranker is configured in this project.
                enable_rerank=False,

                # Return retrieval context instead of generating
                # an answer.
                only_need_context=True,
            ),
        )

        if not isinstance(context_result, dict):
            raise RuntimeError(
                f"{question_id}: unexpected context result type: "
                f"{type(context_result).__name__}"
            )

        if context_result.get("status") == "failure":
            raise RuntimeError(
                f"{question_id}: context retrieval failed: "
                f"{context_result.get('message')}"
            )

        context = context_result.get(
            "llm_response", {}
        ).get(
            "content", ""
        )

        if not context or not context.strip():
            raise RuntimeError(
                f"{question_id}: retrieved context is empty."
            )

        context = context.strip()

        print(
            "Context captured:",
            len(context),
            "characters"
        )

        # References are useful metadata, but RAGAS receives
        # the actual retrieved context separately.
        references = (
            context_result
            .get("data", {})
            .get("references", [])
        )

        print(
            "LightRAG references:",
            len(references)
        )

        # ====================================================
        # STEP 2 — GENERATE ANSWER
        # ====================================================

        print()
        print("[2/2] Generating answer...")

        answer_result = await rag.aquery_llm(
            question,
            param=QueryParam(
                mode="hybrid",
                response_type="Multiple Paragraphs",
                enable_rerank=False,
            ),
        )

        if not isinstance(answer_result, dict):
            raise RuntimeError(
                f"{question_id}: unexpected answer result type: "
                f"{type(answer_result).__name__}"
            )

        if answer_result.get("status") == "failure":
            raise RuntimeError(
                f"{question_id}: answer generation failed: "
                f"{answer_result.get('message')}"
            )

        answer = (
            answer_result
            .get("llm_response", {})
            .get("content", "")
        )

        if not answer or not answer.strip():
            raise RuntimeError(
                f"{question_id}: generated answer is empty."
            )

        answer = answer.strip()

        print(
            "Answer captured:",
            len(answer),
            "characters"
        )

        # ====================================================
        # BUILD RAGAS SAMPLE
        # ====================================================

        sample = {
            "id": question_id,

            # RAGAS-compatible fields
            "user_input": question,
            "response": answer,

            # RAGAS expects a list of retrieved contexts.
            # LightRAG exposes its assembled retrieval context
            # as one complete context string.
            "retrieved_contexts": [
                context
            ],

            "reference": item["reference"],

            # Preserve LightRAG citation metadata for inspection.
            "references": references,
        }

        results.append(sample)

        print()
        print("SUCCESS:", question_id)

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if len(results) != len(QUESTIONS):
        raise RuntimeError(
            f"Expected {len(QUESTIONS)} samples, "
            f"but created {len(results)}."
        )

    for sample in results:

        if not sample["user_input"]:
            raise RuntimeError(
                f"{sample['id']}: missing user_input"
            )

        if not sample["response"]:
            raise RuntimeError(
                f"{sample['id']}: missing response"
            )

        if not sample["retrieved_contexts"]:
            raise RuntimeError(
                f"{sample['id']}: missing retrieved_contexts"
            )

        if not sample["retrieved_contexts"][0].strip():
            raise RuntimeError(
                f"{sample['id']}: retrieved context is empty"
            )

        if not sample["reference"]:
            raise RuntimeError(
                f"{sample['id']}: missing reference"
            )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("RAGAS DATASET CREATED SUCCESSFULLY")
    print("=" * 72)
    print("Dataset      : RAG-001")
    print("Samples      :", len(results))
    print("Output       :", OUTPUT_FILE)
    print("=" * 72)

    print()
    print("Validation:")
    for sample in results:
        print(
            f"{sample['id']}: "
            f"context={len(sample['retrieved_contexts'][0])} chars, "
            f"answer={len(sample['response'])} chars"
        )

    print()
    print("Ready for RAGAS evaluation.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())
