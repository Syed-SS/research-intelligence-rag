# -*- coding: utf-8 -*-

import os
import json
import asyncio
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import (
    openai_complete_if_cache,
    openai_embed,
)
from lightrag.utils import EmbeddingFunc


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

WORKING_DIR = BASE_DIR / "rag_storage_test"

INPUT_FILE = (
    BASE_DIR
    / "ragas_batch_questions_rag002_010.json"
)

OUTPUT_FILE = (
    BASE_DIR
    / "ragas_batch_rag002_010_data.json"
)


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
# LOAD QUESTIONS
# ============================================================

def load_questions():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    questions = data.get(
        "questions",
        []
    )

    if not isinstance(
        questions,
        list,
    ):

        raise RuntimeError(
            "Invalid questions format."
        )

    if len(questions) != 27:

        raise RuntimeError(
            f"Expected 27 questions, "
            f"found {len(questions)}."
        )

    return questions


# ============================================================
# SAVE PROGRESS
# ============================================================

def save_results(results):

    temp_file = OUTPUT_FILE.with_suffix(
        ".partial.json"
    )

    output = {
        "dataset": "RAG-002-RAG-010",
        "total_questions": len(results),
        "questions": results,
    }

    with open(
        temp_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# MAIN
# ============================================================

async def main():

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    questions = load_questions()

    print()
    print("=" * 70)
    print("RAGAS LIGHTRAG DATA PREPARATION")
    print("=" * 70)
    print("Dataset       : RAG-002 -> RAG-010")
    print("Questions     :", len(questions))
    print("LightRAG dir  :", WORKING_DIR)
    print("=" * 70)

    # ========================================================
    # LIGHTRAG
    # ========================================================

    rag = LightRAG(
        working_dir=str(WORKING_DIR),

        llm_model_func=llm_model_func,

        embedding_func=EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model="text-embedding-3-small",
                api_key=api_key,
            ),
        ),

        llm_model_name="gpt-4o-mini",
    )

    await rag.initialize_storages()

    print()
    print("LightRAG loaded successfully.")

    # ========================================================
    # EXISTING PARTIAL RESULTS
    # ========================================================

    partial_file = OUTPUT_FILE.with_suffix(
        ".partial.json"
    )

    existing_results = {}

    if partial_file.exists():

        try:

            with open(
                partial_file,
                "r",
                encoding="utf-8",
            ) as f:

                partial_data = json.load(f)

            for item in partial_data.get(
                "questions",
                [],
            ):

                item_id = item.get(
                    "id"
                )

                if item_id:

                    existing_results[
                        item_id
                    ] = item

            print(
                "Existing progress:",
                len(existing_results),
                "questions",
            )

        except Exception:

            print(
                "WARNING: Partial file could not "
                "be loaded. Starting fresh."
            )

    results = []

    # ========================================================
    # PROCESS QUESTIONS
    # ========================================================

    for index, item in enumerate(
        questions,
        start=1,
    ):

        question_id = item["id"]

        # ----------------------------------------------------
        # RESUME
        # ----------------------------------------------------

        if question_id in existing_results:

            print()
            print(
                f"[{index}/27] {question_id}"
            )

            print(
                "Already completed - skipping."
            )

            results.append(
                existing_results[
                    question_id
                ]
            )

            continue

        question = item["question"]

        reference = item["reference"]

        paper_id = item["paper_id"]

        print()
        print("=" * 70)
        print(
            f"[{index}/27] {question_id}"
        )
        print("=" * 70)

        print(
            "Paper:",
            paper_id,
        )

        print(
            "QUESTION:",
            question,
        )

        # ====================================================
        # RETRIEVE CONTEXT
        # ====================================================

        print()
        print(
            "[1/2] Retrieving context..."
        )

        context_result = await rag.aquery_llm(
            question,
            param=QueryParam(
                mode="hybrid",
                response_type="Multiple Paragraphs",
                only_need_context=True,
                include_references=True,
                enable_rerank=False,
                top_k=40,
                chunk_top_k=20,
            ),
        )

        if not isinstance(
            context_result,
            dict,
        ):

            raise RuntimeError(
                f"{question_id}: invalid "
                "LightRAG context result."
            )

        if context_result.get(
            "status"
        ) == "failure":

            raise RuntimeError(
                f"{question_id}: context retrieval failed: "
                f"{context_result.get('message')}"
            )

        context = context_result.get(
            "llm_response",
            {}
        ).get(
            "content"
        )

        if not context:

            # Some LightRAG versions expose the
            # context through the structured data.
            data = context_result.get(
                "data",
                {}
            )

            context = (
                data.get("context")
                if isinstance(data, dict)
                else None
            )

        if not context:

            raise RuntimeError(
                f"{question_id}: retrieved context is empty."
            )

        metadata = context_result.get(
            "metadata",
            {}
        )

        data = context_result.get(
            "data",
            {}
        )

        references = []

        if isinstance(
            data,
            dict,
        ):

            references = data.get(
                "references",
                []
            )

        print(
            "Context captured:",
            f"{len(context):,}",
            "characters",
        )

        print(
            "LightRAG references:",
            len(references),
        )

        # ====================================================
        # GENERATE ANSWER
        # ====================================================

        print()
        print(
            "[2/2] Generating answer..."
        )

        answer_result = await rag.aquery_llm(
            question,
            param=QueryParam(
                mode="hybrid",
                response_type="Multiple Paragraphs",
                include_references=True,
                enable_rerank=False,
                top_k=40,
                chunk_top_k=20,
            ),
        )

        if not isinstance(
            answer_result,
            dict,
        ):

            raise RuntimeError(
                f"{question_id}: invalid "
                "LightRAG answer result."
            )

        if answer_result.get(
            "status"
        ) == "failure":

            raise RuntimeError(
                f"{question_id}: answer generation failed: "
                f"{answer_result.get('message')}"
            )

        answer = answer_result.get(
            "llm_response",
            {}
        ).get(
            "content"
        )

        if not answer:

            raise RuntimeError(
                f"{question_id}: generated answer is empty."
            )

        answer_data = answer_result.get(
            "data",
            {}
        )

        answer_references = []

        if isinstance(
            answer_data,
            dict,
        ):

            answer_references = (
                answer_data.get(
                    "references",
                    []
                )
            )

        if answer_references:

            references = answer_references

        print(
            "Answer captured:",
            f"{len(answer):,}",
            "characters",
        )

        # ====================================================
        # SAVE ITEM
        # ====================================================

        result_item = {
            "id": question_id,
            "paper_id": paper_id,
            "user_input": question,
            "response": answer,
            "retrieved_contexts": [
                context
            ],
            "reference": reference,
            "source_file": item.get(
                "source_file",
                f"{paper_id}.txt",
            ),
            "source_evidence": item.get(
                "evidence",
                "",
            ),
            "references": references,
            "metadata": metadata,
        }

        results.append(
            result_item
        )

        # Keep results ordered.
        results.sort(
            key=lambda x: x["id"]
        )

        save_results(
            results
        )

        print()
        print(
            f"SUCCESS: {question_id}"
        )

        print(
            "Progress saved:",
            len(results),
            "/ 27",
        )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    results.sort(
        key=lambda x: x["id"]
    )

    if len(results) != 27:

        raise RuntimeError(
            f"Expected 27 results, "
            f"found {len(results)}."
        )

    for item in results:

        required = [
            "id",
            "paper_id",
            "user_input",
            "response",
            "retrieved_contexts",
            "reference",
        ]

        for field in required:

            if field not in item:

                raise RuntimeError(
                    f"{item['id']}: missing {field}"
                )

        if not item["response"].strip():

            raise RuntimeError(
                f"{item['id']}: empty response."
            )

        if not item["retrieved_contexts"]:

            raise RuntimeError(
                f"{item['id']}: empty context."
            )

        if not item["reference"].strip():

            raise RuntimeError(
                f"{item['id']}: empty reference."
            )

    # ========================================================
    # FINAL SAVE
    # ========================================================

    final_output = {
        "dataset": "RAG-002-RAG-010",
        "description": (
            "LightRAG retrieval and answer-generation "
            "dataset prepared for RAGAS evaluation."
        ),
        "papers": [
            "RAG-002",
            "RAG-003",
            "RAG-004",
            "RAG-005",
            "RAG-006",
            "RAG-007",
            "RAG-008",
            "RAG-009",
            "RAG-010",
        ],
        "total_questions": 27,
        "questions": results,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            final_output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Remove partial file after successful completion.
    if partial_file.exists():

        try:
            partial_file.unlink()
        except Exception:
            pass

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("RAGAS DATA PREPARATION COMPLETE")
    print("=" * 70)

    print(
        "Papers    : 9"
    )

    print(
        "Questions : 27"
    )

    print(
        "Contexts  : 27"
    )

    print(
        "Answers   : 27"
    )

    print(
        "Output    :",
        OUTPUT_FILE,
    )

    print("=" * 70)

    print()
    print(
        "Ready for RAGAS evaluation."
    )


if __name__ == "__main__":

    asyncio.run(main())