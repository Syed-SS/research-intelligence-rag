# -*- coding: utf-8 -*-

import os
import json
import math
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace",
    )

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(
        encoding="utf-8",
        errors="replace",
    )
from datasets import Dataset
from ragas import evaluate

# RAGAS 0.4.3 legacy metrics
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

from ragas.llms import llm_factory
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR / "ragas_batch_rag002_010_data.json"
)

OUTPUT_FILE = (
    BASE_DIR / "ragas_batch_rag002_010_results.json"
)

EXPECTED_SAMPLES = 27

MODEL = "gpt-4o-mini"

EMBEDDING_MODEL = "text-embedding-3-small"


# ============================================================
# JSON HELPERS
# ============================================================

def clean_for_json(value):
    """
    Convert NaN / infinity values into JSON-safe None.
    Recursively handles dictionaries and lists.
    """

    if isinstance(value, float):

        if not math.isfinite(value):
            return None

        return value

    if isinstance(value, dict):

        return {
            str(k): clean_for_json(v)
            for k, v in value.items()
        }

    if isinstance(value, list):

        return [
            clean_for_json(v)
            for v in value
        ]

    return value


def atomic_json_write(path, data):
    """
    Write JSON to a temporary file first, then replace
    the destination atomically.

    This prevents a half-written/corrupted JSON file.
    """

    path = Path(path)

    clean_data = clean_for_json(data)

    fd, temp_name = tempfile.mkstemp(
        prefix=path.stem + "_",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )

    try:

        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                clean_data,
                f,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

            f.flush()

            os.fsync(f.fileno())

        os.replace(
            temp_name,
            path,
        )

    except Exception:

        try:
            os.remove(temp_name)
        except OSError:
            pass

        raise


# ============================================================
# LOAD DATASET
# ============================================================

def load_questions():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    questions = data.get(
        "questions",
        [],
    )

    if not isinstance(
        questions,
        list,
    ):

        raise RuntimeError(
            "'questions' must be a list."
        )

    if len(questions) != EXPECTED_SAMPLES:

        raise RuntimeError(
            f"Expected {EXPECTED_SAMPLES} questions, "
            f"found {len(questions)}."
        )

    return questions


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_questions(questions):

    required_fields = [
        "id",
        "user_input",
        "response",
        "retrieved_contexts",
        "reference",
    ]

    for item in questions:

        item_id = item.get(
            "id",
            "UNKNOWN",
        )

        for field in required_fields:

            if field not in item:

                raise RuntimeError(
                    f"{item_id}: missing '{field}'."
                )

        if not str(
            item["user_input"]
        ).strip():

            raise RuntimeError(
                f"{item_id}: empty user_input."
            )

        if not str(
            item["response"]
        ).strip():

            raise RuntimeError(
                f"{item_id}: empty response."
            )

        contexts = item[
            "retrieved_contexts"
        ]

        if not isinstance(
            contexts,
            list,
        ) or not contexts:

            raise RuntimeError(
                f"{item_id}: invalid/empty "
                "retrieved_contexts."
            )

        if not str(
            item["reference"]
        ).strip():

            raise RuntimeError(
                f"{item_id}: empty reference."
            )


# ============================================================
# BUILD RAGAS DATASET
# ============================================================

def build_ragas_dataset(questions):

    rows = []

    for item in questions:

        rows.append(
            {
                "user_input": item[
                    "user_input"
                ],

                "response": item[
                    "response"
                ],

                "retrieved_contexts": item[
                    "retrieved_contexts"
                ],

                "reference": item[
                    "reference"
                ],
            }
        )

    return Dataset.from_list(rows)


# ============================================================
# CREATE EVALUATOR
# ============================================================

def create_evaluator(api_key):

    print()
    print(
        "Testing evaluator authentication..."
    )

    client = OpenAI(
        api_key=api_key
    )

    test = client.chat.completions.create(
        model=MODEL,

        messages=[
            {
                "role": "user",
                "content": "Reply only: OK",
            }
        ],

        max_tokens=5,
    )

    response = (
        test.choices[0]
        .message
        .content
    )

    if response != "OK":

        raise RuntimeError(
            "OpenAI authentication test "
            f"returned: {response}"
        )

    print(
        "Evaluator authentication: OK"
    )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    print()
    print(
        "Creating RAGAS evaluator LLM..."
    )

    evaluator_llm = llm_factory(
        MODEL,
        client=client,
    )

    print(
        "Evaluator LLM:",
        MODEL,
    )

    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    print()
    print(
        "Creating legacy-compatible "
        "OpenAI embeddings..."
    )

    evaluator_embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key,
    )

    print(
        "Evaluator embeddings:",
        EMBEDDING_MODEL,
    )

    # --------------------------------------------------------
    # EMBEDDING TEST
    # --------------------------------------------------------

    print()
    print(
        "Testing embedding interface..."
    )

    test_embedding = (
        evaluator_embeddings.embed_query(
            "RAG evaluation test"
        )
    )

    if not test_embedding:

        raise RuntimeError(
            "Embedding test returned empty result."
        )

    print(
        "Embedding interface: OK"
    )

    print(
        "Embedding dimensions:",
        len(test_embedding),
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    print()
    print(
        "Creating RAGAS metrics..."
    )

    metrics = [
        Faithfulness(
            llm=evaluator_llm
        ),

        AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        ),

        ContextPrecision(
            llm=evaluator_llm
        ),

        ContextRecall(
            llm=evaluator_llm
        ),
    ]

    # --------------------------------------------------------
    # TYPE CHECK
    # --------------------------------------------------------

    from ragas.metrics.base import Metric

    for metric in metrics:

        if not isinstance(
            metric,
            Metric,
        ):

            raise RuntimeError(
                "Invalid RAGAS metric object: "
                f"{type(metric)}"
            )

    print()
    print(
        "Metrics initialized successfully:"
    )

    print(
        "  1. Faithfulness"
    )

    print(
        "  2. Answer Relevancy"
    )

    print(
        "  3. Context Precision"
    )

    print(
        "  4. Context Recall"
    )

    return (
        client,
        evaluator_llm,
        evaluator_embeddings,
        metrics,
    )


# ============================================================
# SINGLE QUESTION DIAGNOSTIC
# ============================================================

def run_diagnostic(
    questions,
    evaluator_llm,
    evaluator_embeddings,
    metrics,
):

    print()
    print("=" * 70)
    print(
        "RAGAS SINGLE-QUESTION DIAGNOSTIC"
    )
    print("=" * 70)

    first = questions[0]

    print(
        "Question:",
        first["id"],
    )

    print(
        first["user_input"]
    )

    diagnostic_dataset = Dataset.from_list(
        [
            {
                "user_input": first[
                    "user_input"
                ],

                "response": first[
                    "response"
                ],

                "retrieved_contexts": first[
                    "retrieved_contexts"
                ],

                "reference": first[
                    "reference"
                ],
            }
        ]
    )

    print()
    print(
        "Running all 4 metrics on one sample..."
    )

    try:

        result = evaluate(
            dataset=diagnostic_dataset,

            metrics=metrics,

            llm=evaluator_llm,

            embeddings=evaluator_embeddings,

            raise_exceptions=True,

            show_progress=False,
        )

    except Exception as exc:

        print()
        print("=" * 70)
        print(
            "DIAGNOSTIC FAILED"
        )
        print("=" * 70)

        print(
            type(exc).__name__
        )

        print(
            str(exc)
        )

        print()
        print(
            "Full batch evaluation will NOT start."
        )

        print("=" * 70)

        return False

    df = result.to_pandas()

    print()
    print(
        "Diagnostic result:"
    )

    print(
        df.to_string(
            index=False
        )
    )

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    failed = []

    for metric_name in metric_names:

        if metric_name not in df.columns:

            failed.append(
                metric_name
            )

            continue

        value = df.iloc[0][
            metric_name
        ]

        if (
            isinstance(
                value,
                float,
            )
            and math.isnan(value)
        ):

            failed.append(
                metric_name
            )

    print()

    if failed:

        print(
            "Diagnostic produced NaN for:"
        )

        for name in failed:

            print(
                "  -",
                name,
            )

        print()
        print(
            "Full batch evaluation will NOT start."
        )

        return False

    print(
        "DIAGNOSTIC: PASS"
    )

    print(
        "All 4 metrics returned valid scores."
    )

    return True


# ============================================================
# FULL EVALUATION
# ============================================================

def run_full_evaluation(
    dataset,
    evaluator_llm,
    evaluator_embeddings,
    metrics,
):

    print()
    print("=" * 70)
    print(
        "STARTING FULL RAGAS EVALUATION"
    )
    print("=" * 70)

    print(
        "Questions:",
        len(dataset),
    )

    print(
        "Metrics:",
        len(metrics),
    )

    print(
        "Total evaluations:",
        len(dataset)
        * len(metrics),
    )

    print()

    result = evaluate(
        dataset=dataset,

        metrics=metrics,

        llm=evaluator_llm,

        embeddings=evaluator_embeddings,

        #
        # Do not crash the entire 27-question
        # evaluation because one row fails.
        #
        raise_exceptions=False,

        show_progress=True,
    )

    return result


# ============================================================
# ANALYZE RESULTS
# ============================================================

def analyze_results(result_df):

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    overall = {}

    nan_counts = {}

    for metric_name in metric_names:

        if metric_name not in result_df.columns:

            overall[
                metric_name
            ] = None

            nan_counts[
                metric_name
            ] = len(result_df)

            continue

        values = result_df[
            metric_name
        ]

        valid_values = []

        nan_count = 0

        for value in values:

            if (
                isinstance(
                    value,
                    float,
                )
                and math.isnan(value)
            ):

                nan_count += 1

            elif value is None:

                nan_count += 1

            else:

                valid_values.append(
                    float(value)
                )

        nan_counts[
            metric_name
        ] = nan_count

        if valid_values:

            overall[
                metric_name
            ] = sum(
                valid_values
            ) / len(
                valid_values
            )

        else:

            overall[
                metric_name
            ] = None

    return (
        overall,
        nan_counts,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "RAGAS BATCH EVALUATION"
    )
    print("=" * 70)

    print(
        "RAGAS version : 0.4.3"
    )

    print(
        "Dataset       : RAG-002 -> RAG-010"
    )

    print(
        "Expected      : 27 samples"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # API KEY
    # --------------------------------------------------------

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print()
    print(
        "Loading evaluation dataset..."
    )

    questions = load_questions()

    validate_questions(
        questions
    )

    print(
        "Input dataset validation: OK"
    )

    print(
        "Questions loaded:",
        len(questions),
    )

    # --------------------------------------------------------
    # BUILD DATASET
    # --------------------------------------------------------

    dataset = build_ragas_dataset(
        questions
    )

    # --------------------------------------------------------
    # EVALUATOR
    # --------------------------------------------------------

    (
        client,
        evaluator_llm,
        evaluator_embeddings,
        metrics,
    ) = create_evaluator(
        api_key
    )

    # --------------------------------------------------------
    # DIAGNOSTIC
    # --------------------------------------------------------

    diagnostic_passed = run_diagnostic(
        questions,

        evaluator_llm,

        evaluator_embeddings,

        metrics,
    )

    if not diagnostic_passed:

        raise RuntimeError(
            "RAGAS diagnostic failed. "
            "Full evaluation was intentionally stopped."
        )

    # --------------------------------------------------------
    # FULL EVALUATION
    # --------------------------------------------------------

    result = run_full_evaluation(
        dataset,

        evaluator_llm,

        evaluator_embeddings,

        metrics,
    )

    result_df = result.to_pandas()

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "RAGAS RESULTS"
    )
    print("=" * 70)

    print(
        result_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SCORES
    # --------------------------------------------------------

    (
        overall_scores,
        nan_counts,
    ) = analyze_results(
        result_df
    )

    print()
    print("=" * 70)
    print(
        "OVERALL SCORES"
    )
    print("=" * 70)

    for name, value in overall_scores.items():

        if value is None:

            print(
                f"{name:<20}: NaN"
            )

        else:

            print(
                f"{name:<20}: "
                f"{value:.4f}"
            )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "METRIC VALIDATION"
    )
    print("=" * 70)

    for name, count in nan_counts.items():

        print(
            f"{name:<20}: "
            f"{count} NaN / "
            f"{len(result_df)}"
        )

    # --------------------------------------------------------
    # BUILD OUTPUT
    # --------------------------------------------------------

    output = {
        "dataset": "RAG-002-RAG-010",

        "ragas_version": "0.4.3",

        "model": MODEL,

        "embedding_model": EMBEDDING_MODEL,

        "samples": len(dataset),

        "total_evaluations": (
            len(dataset)
            * len(metrics)
        ),

        "metrics": [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ],

        "overall_scores": overall_scores,

        "nan_counts": nan_counts,

        "results": result_df.to_dict(
            orient="records"
        ),
    }

    # --------------------------------------------------------
    # SAFE JSON WRITE
    # --------------------------------------------------------

    atomic_json_write(
        OUTPUT_FILE,
        output,
    )

    # --------------------------------------------------------
    # VERIFY JSON
    # --------------------------------------------------------

    print()
    print(
        "Verifying saved JSON..."
    )

    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        verified = json.load(f)

    if (
        verified.get("samples")
        != EXPECTED_SAMPLES
    ):

        raise RuntimeError(
            "Saved JSON verification failed: "
            "sample count mismatch."
        )

    print(
        "Saved JSON validation: OK"
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "RAGAS EVALUATION COMPLETE"
    )
    print("=" * 70)

    print(
        "Dataset : RAG-002 -> RAG-010"
    )

    print(
        "Samples :",
        len(dataset),
    )

    print(
        "Metrics :",
        len(metrics),
    )

    print(
        "Results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()