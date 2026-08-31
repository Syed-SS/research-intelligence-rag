import os
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from datasets import Dataset
from ragas import evaluate

# RAGAS 0.4.3 compatible classic metrics
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_openai import OpenAIEmbeddings
from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "ragas_rag001_data.json"
OUTPUT_FILE = BASE_DIR / "ragas_rag001_results.json"

RAGAS_VERSION = "0.4.3"
EVALUATION_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

EXPECTED_SAMPLES = 3


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):
    if value is None:
        return ""

    return str(value).strip()


def normalize_contexts(contexts):
    """
    RAGAS expects retrieved_contexts as list[str].
    Normalize the LightRAG-generated data accordingly.
    """

    if contexts is None:
        return []

    if isinstance(contexts, str):
        contexts = [contexts]

    normalized = []

    for context in contexts:

        if context is None:
            continue

        if isinstance(context, dict):

            value = (
                context.get("content")
                or context.get("text")
                or context.get("chunk")
                or context.get("context")
            )

            if value is None:
                value = json.dumps(
                    context,
                    ensure_ascii=False,
                )

            context = value

        context = clean_text(context)

        if context:
            normalized.append(context)

    return normalized


def validate_data(data):

    if not isinstance(data, list):
        raise RuntimeError(
            "RAGAS input must be a JSON list."
        )

    if len(data) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_SAMPLES} samples, "
            f"found {len(data)}."
        )

    required_fields = [
        "user_input",
        "response",
        "retrieved_contexts",
        "reference",
    ]

    for item in data:

        item_id = item.get(
            "id",
            "UNKNOWN",
        )

        for field in required_fields:

            if field not in item:
                raise RuntimeError(
                    f"{item_id}: missing '{field}'."
                )

        if not clean_text(
            item["user_input"]
        ):
            raise RuntimeError(
                f"{item_id}: empty user_input."
            )

        if not clean_text(
            item["response"]
        ):
            raise RuntimeError(
                f"{item_id}: empty response."
            )

        contexts = normalize_contexts(
            item["retrieved_contexts"]
        )

        if not contexts:
            raise RuntimeError(
                f"{item_id}: empty retrieved_contexts."
            )

        if not clean_text(
            item["reference"]
        ):
            raise RuntimeError(
                f"{item_id}: empty reference."
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("RAGAS EVALUATION")
    print("=" * 70)

    print(
        f"RAGAS version : {RAGAS_VERSION}"
    )

    print(
        "Dataset       : RAG-001"
    )

    print(
        f"Samples       : {EXPECTED_SAMPLES}"
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
    # LOAD INPUT
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    validate_data(data)

    # --------------------------------------------------------
    # BUILD DATASET
    # --------------------------------------------------------

    ragas_data = []

    for item in data:

        ragas_data.append(
            {
                "user_input": clean_text(
                    item["user_input"]
                ),

                "response": clean_text(
                    item["response"]
                ),

                "retrieved_contexts":
                    normalize_contexts(
                        item["retrieved_contexts"]
                    ),

                "reference": clean_text(
                    item["reference"]
                ),
            }
        )

    dataset = Dataset.from_list(
        ragas_data
    )

    print()
    print(
        "Input dataset validation: OK"
    )

    # --------------------------------------------------------
    # OPENAI CLIENT
    # --------------------------------------------------------

    print()
    print(
        "Testing evaluator authentication..."
    )

    client = OpenAI(
        api_key=api_key
    )

    test = client.chat.completions.create(
        model=EVALUATION_MODEL,
        messages=[
            {
                "role": "user",
                "content": "Reply only: OK",
            }
        ],
        max_tokens=5,
    )

    print(
        "Evaluator authentication:",
        test.choices[0].message.content,
    )

    # --------------------------------------------------------
    # RAGAS LLM
    # --------------------------------------------------------

    print()
    print(
        "Creating RAGAS evaluator LLM..."
    )

    evaluator_llm = llm_factory(
        EVALUATION_MODEL,
        client=client,
    )

    print(
        "Evaluator LLM:",
        EVALUATION_MODEL,
    )

    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    print()
    print(
        "Creating evaluator embeddings..."
    )

    base_embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key,
    )

    evaluator_embeddings = (
        LangchainEmbeddingsWrapper(
            base_embeddings
        )
    )

    print(
        "Evaluator embeddings:",
        EMBEDDING_MODEL,
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    print()
    print(
        "Creating RAGAS metrics..."
    )

    faithfulness = Faithfulness(
        llm=evaluator_llm
    )

    answer_relevancy = AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    context_precision = ContextPrecision(
        llm=evaluator_llm
    )

    context_recall = ContextRecall(
        llm=evaluator_llm
    )

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

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

    # --------------------------------------------------------
    # FINAL TYPE CHECK
    # --------------------------------------------------------

    from ragas.metrics.base import Metric

    print()
    print(
        "Metric compatibility check:"
    )

    for metric in metrics:

        print(
            f"  {metric.name}: "
            f"{isinstance(metric, Metric)}"
        )

        if not isinstance(
            metric,
            Metric,
        ):
            raise RuntimeError(
                f"{metric.name} is not a "
                f"RAGAS Metric object."
            )

    print(
        "Metric compatibility: OK"
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "STARTING RAGAS EVALUATION"
    )
    print("=" * 70)

    print(
        f"Questions         : {len(dataset)}"
    )

    print(
        f"Metrics           : {len(metrics)}"
    )

    print(
        f"Total evaluations : "
        f"{len(dataset) * len(metrics)}"
    )

    print(
        "Batch size        : 1"
    )

    print()

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        raise_exceptions=False,
        show_progress=True,
        batch_size=1,
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    result_df = result.to_pandas()

    print()
    print("=" * 70)
    print("RAGAS RESULTS")
    print("=" * 70)

    print()

    print(
        result_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # OVERALL SCORES
    # --------------------------------------------------------

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    overall_scores = {}

    for metric_name in metric_names:

        if metric_name not in result_df.columns:

            overall_scores[
                metric_name
            ] = None

            continue

        values = result_df[
            metric_name
        ].dropna()

        if len(values) == 0:

            overall_scores[
                metric_name
            ] = None

        else:

            overall_scores[
                metric_name
            ] = float(
                values.mean()
            )

    print()
    print("=" * 70)
    print("OVERALL SCORES")
    print("=" * 70)

    for metric_name in metric_names:

        score = overall_scores[
            metric_name
        ]

        if score is None:

            print(
                f"{metric_name:20s}: NaN"
            )

        else:

            print(
                f"{metric_name:20s}: "
                f"{score:.4f}"
            )

    # --------------------------------------------------------
    # PER QUESTION
    # --------------------------------------------------------

    per_question_results = []

    for index, row in result_df.iterrows():

        per_question_results.append(
            {
                "question_number":
                    index + 1,

                "user_input":
                    row.get(
                        "user_input"
                    ),

                "faithfulness":
                    row.get(
                        "faithfulness"
                    ),

                "answer_relevancy":
                    row.get(
                        "answer_relevancy"
                    ),

                "context_precision":
                    row.get(
                        "context_precision"
                    ),

                "context_recall":
                    row.get(
                        "context_recall"
                    ),
            }
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    output = {
        "dataset": "RAG-001",

        "ragas_version":
            RAGAS_VERSION,

        "samples":
            len(data),

        "evaluation_model":
            EVALUATION_MODEL,

        "embedding_model":
            EMBEDDING_MODEL,

        "metrics":
            metric_names,

        "overall_scores":
            overall_scores,

        "per_question_results":
            per_question_results,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
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
        "Results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print("=" * 70)


if __name__ == "__main__":
    main()