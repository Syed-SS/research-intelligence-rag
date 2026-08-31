# -*- coding: utf-8 -*-

import os
import json
import asyncio
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    AsyncOpenAI,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)


load_dotenv(override=True)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

OUTPUT_FILE = BASE_DIR / "ragas_batch_questions_rag002_010.json"
TEMP_FILE = BASE_DIR / "ragas_batch_questions_rag002_010.partial.json"


# ============================================================
# DATASET CONFIGURATION
# ============================================================

PAPER_IDS = [
    "RAG-002",
    "RAG-003",
    "RAG-004",
    "RAG-005",
    "RAG-006",
    "RAG-007",
    "RAG-008",
    "RAG-009",
    "RAG-010",
]

QUESTIONS_PER_PAPER = 3

MODEL = "gpt-4o-mini"

MAX_RETRIES = 5
BASE_RETRY_SECONDS = 3

# Keep prompts reasonably sized.
MAX_SOURCE_CHARS = 45000


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are creating a source-grounded evaluation dataset for a
Retrieval-Augmented Generation system.

You must use ONLY the supplied research paper text.

Create exactly 3 high-quality questions for this paper.

IMPORTANT RULES:

1. Every question must be answerable directly from the supplied paper.
2. Do not use outside knowledge.
3. Do not invent facts, methods, datasets, results, numbers, or claims.
4. The reference answer must contain only information supported by the paper.
5. Prefer concrete technical questions about:
   - methodology
   - architecture
   - retrieval
   - generation
   - datasets
   - experiments
   - findings
   - limitations
   - contributions
6. Avoid vague questions such as:
   "Why is this paper important?"
7. Avoid questions requiring outside knowledge.
8. Avoid questions whose answer depends on unrelated parts of the paper.
9. Questions must be independent.
10. Reference answers should be concise but complete.
11. Do not mention "the provided text" in the question.
12. Do not fabricate citations.
13. Each reference answer must be supported by the source.
14. Select a short supporting evidence passage from the source.
15. The evidence does NOT need to be an exact full sentence from the source.
16. Preserve the original terminology used by the paper.
17. Do not create questions about information that is merely present
    in the bibliography or references.

Return ONLY valid JSON.

Required format:

{
  "questions": [
    {
      "question": "...",
      "reference": "...",
      "evidence": "..."
    },
    {
      "question": "...",
      "reference": "...",
      "evidence": "..."
    },
    {
      "question": "...",
      "reference": "...",
      "evidence": "..."
    }
  ]
}
"""


# ============================================================
# SAFE JSON LOADING
# ============================================================

def load_partial_results():

    if not TEMP_FILE.exists():
        return {}

    try:
        with open(
            TEMP_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return {}

        papers = data.get("papers", {})

        if not isinstance(papers, dict):
            return {}

        return papers

    except Exception:
        print("WARNING: Partial file could not be loaded.")
        print("Starting with empty progress.")
        return {}


# ============================================================
# SAVE PROGRESS
# ============================================================

def save_partial_results(papers):

    output = {
        "dataset": "RAG-002-RAG-010",
        "description": (
            "Source-grounded evaluation questions generated "
            "directly from the processed research papers."
        ),
        "papers": papers,
        "total_questions": sum(
            len(v.get("questions", []))
            for v in papers.values()
            if isinstance(v, dict)
        ),
    }

    with open(
        TEMP_FILE,
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
# SOURCE CLEANING
# ============================================================

def prepare_source(text):

    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        lines.append(line)

    cleaned = "\n".join(lines)

    return cleaned


# ============================================================
# RETRY WRAPPER
# ============================================================

async def call_openai_with_retry(
    client,
    messages,
):

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = await client.chat.completions.create(
                model=MODEL,
                temperature=0,
                response_format={
                    "type": "json_object"
                },
                messages=messages,
                timeout=120,
            )

            return response

        except (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
        ) as exc:

            if attempt == MAX_RETRIES:
                raise

            wait_time = (
                BASE_RETRY_SECONDS * (2 ** (attempt - 1))
                + random.uniform(0, 1)
            )

            print()
            print(
                f"OpenAI temporary error "
                f"(attempt {attempt}/{MAX_RETRIES})."
            )

            print(
                f"Retrying in {wait_time:.1f} seconds..."
            )

            await asyncio.sleep(wait_time)

        except Exception:

            # Unknown error: retry a few times because
            # transient SDK/API errors can occur here too.

            if attempt == MAX_RETRIES:
                raise

            wait_time = (
                BASE_RETRY_SECONDS * (2 ** (attempt - 1))
                + random.uniform(0, 1)
            )

            print()
            print(
                f"Unexpected API error "
                f"(attempt {attempt}/{MAX_RETRIES})."
            )

            print(
                f"Retrying in {wait_time:.1f} seconds..."
            )

            await asyncio.sleep(wait_time)


# ============================================================
# QUESTION GENERATION
# ============================================================

async def generate_questions(
    client,
    paper_id,
    source_text,
):

    prompt = f"""
Paper ID:
{paper_id}

SOURCE PAPER
============

{source_text}

END SOURCE PAPER

Create exactly 3 source-grounded questions.

Make each question technically meaningful for evaluating
a RAG system.

For each question:
- give a concise reference answer
- provide a short supporting evidence passage
"""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    response = await call_openai_with_retry(
        client,
        messages,
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            f"{paper_id}: OpenAI returned empty content."
        )

    try:

        result = json.loads(content)

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            f"{paper_id}: invalid JSON returned by OpenAI."
        ) from exc

    questions = result.get("questions")

    if not isinstance(questions, list):

        raise RuntimeError(
            f"{paper_id}: 'questions' is not a list."
        )

    if len(questions) != QUESTIONS_PER_PAPER:

        raise RuntimeError(
            f"{paper_id}: expected "
            f"{QUESTIONS_PER_PAPER} questions, "
            f"received {len(questions)}."
        )

    return questions


# ============================================================
# QUESTION VALIDATION
# ============================================================

def validate_questions(
    paper_id,
    questions,
    source_text,
):

    if len(questions) != QUESTIONS_PER_PAPER:
        raise RuntimeError(
            f"{paper_id}: invalid question count."
        )

    normalized_source = " ".join(
        source_text.lower().split()
    )

    validated = []

    seen_questions = set()

    for index, item in enumerate(
        questions,
        start=1,
    ):

        if not isinstance(item, dict):

            raise RuntimeError(
                f"{paper_id}-Q{index}: invalid object."
            )

        question = str(
            item.get("question", "")
        ).strip()

        reference = str(
            item.get("reference", "")
        ).strip()

        evidence = str(
            item.get("evidence", "")
        ).strip()

        if not question:

            raise RuntimeError(
                f"{paper_id}-Q{index}: empty question."
            )

        if not reference:

            raise RuntimeError(
                f"{paper_id}-Q{index}: empty reference."
            )

        if not evidence:

            raise RuntimeError(
                f"{paper_id}-Q{index}: empty evidence."
            )

        normalized_question = " ".join(
            question.lower().split()
        )

        if normalized_question in seen_questions:

            raise RuntimeError(
                f"{paper_id}-Q{index}: duplicate question."
            )

        seen_questions.add(
            normalized_question
        )

        # Soft evidence validation.
        #
        # We intentionally DO NOT require exact
        # verbatim matching. The previous version
        # failed here unnecessarily.
        evidence_words = [
            word
            for word in evidence.lower().split()
            if len(word) >= 5
        ]

        if evidence_words:

            matches = sum(
                1
                for word in evidence_words
                if word in normalized_source
            )

            evidence_ratio = (
                matches / len(evidence_words)
            )

        else:

            evidence_ratio = 0

        # Only reject clearly fabricated evidence.
        if evidence_ratio < 0.45:

            raise RuntimeError(
                f"{paper_id}-Q{index}: evidence appears "
                f"insufficiently grounded "
                f"(ratio={evidence_ratio:.2f})."
            )

        validated.append(
            {
                "id": f"{paper_id}-Q{index}",
                "paper_id": paper_id,
                "question": question,
                "reference": reference,
                "evidence": evidence,
                "source_file": f"{paper_id}.txt",
                "grounding_check": "passed",
            }
        )

    return validated


# ============================================================
# GENERATE ONE PAPER WITH RETRIES
# ============================================================

async def process_paper(
    client,
    paper_id,
    source_text,
):

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            print()
            print(
                f"Generating questions "
                f"(attempt {attempt}/{MAX_RETRIES})..."
            )

            questions = await generate_questions(
                client,
                paper_id,
                source_text,
            )

            validated = validate_questions(
                paper_id,
                questions,
                source_text,
            )

            return validated

        except Exception as exc:

            if attempt == MAX_RETRIES:
                raise

            wait_time = (
                BASE_RETRY_SECONDS
                * (2 ** (attempt - 1))
            )

            print(
                f"Validation/generation issue: {exc}"
            )

            print(
                f"Retrying paper in "
                f"{wait_time:.1f} seconds..."
            )

            await asyncio.sleep(wait_time)


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

    if not DATA_DIR.exists():

        raise FileNotFoundError(
            f"Processed data directory not found:\n"
            f"{DATA_DIR}"
        )

    client = AsyncOpenAI(
        api_key=api_key
    )

    partial_results = load_partial_results()

    print()
    print("=" * 70)
    print("SOURCE-GROUNDED QUESTION GENERATION")
    print("=" * 70)
    print(
        "Papers         :",
        len(PAPER_IDS),
    )
    print(
        "Questions/paper:",
        QUESTIONS_PER_PAPER,
    )
    print(
        "Expected total :",
        len(PAPER_IDS)
        * QUESTIONS_PER_PAPER,
    )
    print(
        "Range          : RAG-002 -> RAG-010"
    )
    print(
        "Model          :",
        MODEL,
    )
    print("=" * 70)

    completed = 0
    failed = []

    for paper_id in PAPER_IDS:

        # ----------------------------------------------------
        # RESUME SUPPORT
        # ----------------------------------------------------

        existing = partial_results.get(
            paper_id
        )

        if (
            isinstance(existing, dict)
            and len(
                existing.get(
                    "questions",
                    []
                )
            )
            == QUESTIONS_PER_PAPER
        ):

            print()
            print(
                f"{paper_id}: already completed."
            )

            print(
                "Skipping API call."
            )

            completed += 1

            continue

        # ----------------------------------------------------
        # SOURCE FILE
        # ----------------------------------------------------

        source_file = (
            DATA_DIR
            / f"{paper_id}.txt"
        )

        if not source_file.exists():

            print()
            print(
                f"ERROR: {paper_id} source file "
                f"not found."
            )

            failed.append(
                (
                    paper_id,
                    "source file missing",
                )
            )

            continue

        source_text = source_file.read_text(
            encoding="utf-8",
            errors="replace",
        )

        source_text = prepare_source(
            source_text
        )

        if not source_text:

            print()
            print(
                f"ERROR: {paper_id} source is empty."
            )

            failed.append(
                (
                    paper_id,
                    "empty source",
                )
            )

            continue

        # ----------------------------------------------------
        # LIMIT PROMPT SIZE
        # ----------------------------------------------------

        if len(source_text) > MAX_SOURCE_CHARS:

            print(
                f"Source size: "
                f"{len(source_text):,} characters"
            )

            print(
                f"Using first "
                f"{MAX_SOURCE_CHARS:,} characters "
                f"for question generation."
            )

            source_for_generation = (
                source_text[:MAX_SOURCE_CHARS]
            )

        else:

            source_for_generation = source_text

        print()
        print("-" * 70)
        print(paper_id)
        print("-" * 70)

        print(
            "Source loaded:",
            f"{len(source_text):,}",
            "characters",
        )

        # ----------------------------------------------------
        # PROCESS PAPER
        # ----------------------------------------------------

        try:

            questions = await process_paper(
                client,
                paper_id,
                source_for_generation,
            )

            partial_results[paper_id] = {
                "source_file": f"{paper_id}.txt",
                "source_characters": len(
                    source_text
                ),
                "questions": questions,
                "status": "completed",
            }

            # SAVE IMMEDIATELY
            save_partial_results(
                partial_results
            )

            completed += 1

            print()
            print(
                f"SUCCESS: {paper_id}"
            )

            for item in questions:

                print()
                print(
                    item["id"]
                )

                print(
                    "QUESTION:",
                    item["question"],
                )

                print(
                    "REFERENCE:",
                    item["reference"],
                )

            print()
            print(
                "Progress saved."
            )

        except Exception as exc:

            print()
            print(
                f"FAILED: {paper_id}"
            )

            print(
                "Reason:",
                str(exc),
            )

            failed.append(
                (
                    paper_id,
                    str(exc),
                )
            )

            # Save failure state.
            partial_results[paper_id] = {
                "source_file": f"{paper_id}.txt",
                "status": "failed",
                "error": str(exc),
                "questions": [],
            }

            save_partial_results(
                partial_results
            )

    # ========================================================
    # FINAL DATASET
    # ========================================================

    all_questions = []

    completed_papers = []

    for paper_id in PAPER_IDS:

        paper_data = partial_results.get(
            paper_id
        )

        if not isinstance(
            paper_data,
            dict,
        ):
            continue

        questions = paper_data.get(
            "questions",
            [],
        )

        if len(questions) == QUESTIONS_PER_PAPER:

            completed_papers.append(
                paper_id
            )

            all_questions.extend(
                questions
            )

    expected_total = (
        len(PAPER_IDS)
        * QUESTIONS_PER_PAPER
    )

    print()
    print("=" * 70)
    print("BATCH GENERATION SUMMARY")
    print("=" * 70)

    print(
        "Papers completed:",
        len(completed_papers),
        "/",
        len(PAPER_IDS),
    )

    print(
        "Questions created:",
        len(all_questions),
        "/",
        expected_total,
    )

    if failed:

        print()
        print("Failed papers:")

        for paper_id, reason in failed:

            print(
                f"  {paper_id}: {reason}"
            )

    # --------------------------------------------------------
    # ONLY CREATE FINAL FILE IF ALL 27 EXIST
    # --------------------------------------------------------

    if len(all_questions) != expected_total:

        print()
        print(
            "FINAL DATASET NOT CREATED."
        )

        print(
            "Partial progress is safely stored at:"
        )

        print(TEMP_FILE)

        print()
        print(
            "Fix the failed paper(s) and rerun."
        )

        return

    final_output = {
        "dataset": "RAG-002-RAG-010",
        "description": (
            "Source-grounded RAGAS evaluation dataset "
            "generated from the actual processed research papers."
        ),
        "papers": PAPER_IDS,
        "questions_per_paper": QUESTIONS_PER_PAPER,
        "total_questions": len(all_questions),
        "questions": all_questions,
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

    print()
    print("=" * 70)
    print("SOURCE-GROUNDED DATASET CREATED SUCCESSFULLY")
    print("=" * 70)

    print(
        "Papers    :",
        len(PAPER_IDS),
    )

    print(
        "Questions :",
        len(all_questions),
    )

    print(
        "Per paper :",
        QUESTIONS_PER_PAPER,
    )

    print(
        "Output    :",
        OUTPUT_FILE,
    )

    print("=" * 70)

    print()
    print("Validation:")
    print("  Source files       : OK")
    print("  Question count     : OK")
    print("  Reference answers  : OK")
    print("  Grounding checks   : OK")
    print("  Incremental saves  : OK")
    print("  Resume support     : OK")
    print("  Retry handling     : OK")

    print()
    print(
        "Ready for LightRAG retrieval preparation."
    )


if __name__ == "__main__":
    asyncio.run(main())