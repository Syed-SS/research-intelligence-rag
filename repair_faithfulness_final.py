import json
import os
import asyncio
from copy import deepcopy

from dotenv import load_dotenv
load_dotenv(override=True)

from datasets import Dataset
from openai import OpenAI
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness


INPUT_RESULTS = "ragas_batch_rag002_010_results.json"
FAITHFULNESS_DATA = "faithfulness_evaluation_data.json"
OUTPUT_RESULTS = "ragas_batch_rag002_010_faithfulness_fixed.json"

MODEL = "gpt-4o-mini"
STATEMENTS_PER_BATCH = 4


class BatchedFaithfulness(Faithfulness):

    async def _ascore(self, row, callbacks):

        statements_output = await self._create_statements(
            row,
            callbacks
        )

        statements = statements_output.statements

        if not statements:
            return float("nan")

        all_verdicts = []

        for start in range(
            0,
            len(statements),
            STATEMENTS_PER_BATCH
        ):
            batch = statements[
                start:start + STATEMENTS_PER_BATCH
            ]

            verdict_output = await self._create_verdicts(
                row,
                batch,
                callbacks
            )

            all_verdicts.extend(
                verdict_output.statements
            )

        if not all_verdicts:
            return float("nan")

        faithful = sum(
            1
            for verdict in all_verdicts
            if verdict.verdict
        )

        return faithful / len(all_verdicts)


def is_nan(value):
    return value is None or (
        isinstance(value, float) and value != value
    )


def main():

    print("=" * 80)
    print("FINAL FAITHFULNESS REPAIR")
    print("=" * 80)

    # ---------------------------------------------------------
    # Load existing results
    # ---------------------------------------------------------

    with open(
        INPUT_RESULTS,
        "r",
        encoding="utf-8"
    ) as f:
        original = json.load(f)

    # ---------------------------------------------------------
    # Load compact Faithfulness contexts
    # ---------------------------------------------------------

    with open(
        FAITHFULNESS_DATA,
        "r",
        encoding="utf-8"
    ) as f:
        faith_data = json.load(f)

    faith_rows = {
        row["id"]: row
        for row in faith_data["questions"]
    }

    results = deepcopy(
        original["results"]
    )

    failed_indexes = [
        i
        for i, row in enumerate(results)
        if is_nan(row.get("faithfulness"))
    ]

    successful_indexes = [
        i
        for i, row in enumerate(results)
        if not is_nan(row.get("faithfulness"))
    ]

    print()
    print("Total rows              :", len(results))
    print("Existing successful    :", len(successful_indexes))
    print("Faithfulness NaN rows  :", len(failed_indexes))
    print()

    if not failed_indexes:
        print("No Faithfulness NaN rows found.")
        return

    # ---------------------------------------------------------
    # Create evaluator
    # ---------------------------------------------------------

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    client = OpenAI(
        api_key=api_key
    )

    evaluator_llm = llm_factory(
        MODEL,
        client=client
    )

    metric = BatchedFaithfulness(
        llm=evaluator_llm,
        max_retries=1
    )

    # ---------------------------------------------------------
    # Evaluate ONLY the NaN rows
    # ---------------------------------------------------------

    print("=" * 80)
    print("EVALUATING ONLY FAITHFULNESS NaN ROWS")
    print("=" * 80)

    for position, index in enumerate(
        failed_indexes,
        start=1
    ):

        result_row = results[index]
        row_id = result_row.get("id")

        # Match by user_input if id was not retained
        if row_id and row_id in faith_rows:
            source_row = faith_rows[row_id]
        else:
            source_row = next(
                r for r in faith_data["questions"]
                if r["user_input"]
                == result_row["user_input"]
            )

        contexts = source_row.get(
            "faithfulness_contexts",
            []
        )

        if not contexts:
            print(
                f"[{position}/{len(failed_indexes)}] "
                f"{row_id} -> NO CONTEXT, SKIPPED"
            )
            continue

        dataset = Dataset.from_list(
            [{
                "user_input":
                    result_row["user_input"],

                "response":
                    result_row["response"],

                "retrieved_contexts":
                    contexts
            }]
        )

        print(
            f"[{position}/{len(failed_indexes)}] "
            f"{row_id or 'unknown'} -> evaluating..."
        )

        try:

            evaluation = await_evaluate(
                dataset,
                metric
            )

            df = evaluation.to_pandas()

            score = df.iloc[0][
                "faithfulness"
            ]

            if not is_nan(score):

                result_row[
                    "faithfulness"
                ] = float(score)

                print(
                    f"    SUCCESS: {float(score):.4f}"
                )

            else:

                print(
                    "    RESULT: NaN"
                )

        except Exception as exc:

            print(
                f"    ERROR: {type(exc).__name__}: {exc}"
            )

    # ---------------------------------------------------------
    # Recalculate Faithfulness only
    # ---------------------------------------------------------

    faith_scores = [
        r["faithfulness"]
        for r in results
        if not is_nan(
            r.get("faithfulness")
        )
    ]

    final_nan_count = sum(
        1
        for r in results
        if is_nan(
            r.get("faithfulness")
        )
    )

    final_average = (
        sum(faith_scores)
        / len(faith_scores)
        if faith_scores
        else None
    )

    # ---------------------------------------------------------
    # Preserve all other existing result information
    # ---------------------------------------------------------

    output = deepcopy(original)

    output["results"] = results

    output["overall_scores"][
        "faithfulness"
    ] = final_average

    output["nan_counts"][
        "faithfulness"
    ] = final_nan_count

    output[
        "faithfulness_repair"
    ] = {
        "method":
            "Batched Faithfulness evaluation",
        "model":
            MODEL,
        "statements_per_batch":
            STATEMENTS_PER_BATCH,
        "original_nan_rows":
            len(failed_indexes),
        "remaining_nan_rows":
            final_nan_count
    }

    # ---------------------------------------------------------
    # Save NEW file
    # ---------------------------------------------------------

    with open(
        OUTPUT_RESULTS,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("=" * 80)
    print("FINAL REPAIR SUMMARY")
    print("=" * 80)

    print(
        "Faithfulness evaluated :",
        len(failed_indexes)
    )

    print(
        "Faithfulness NaN left  :",
        final_nan_count
    )

    print(
        "Faithfulness average   :",
        final_average
    )

    print()
    print(
        "NEW OUTPUT:"
    )
    print(
        OUTPUT_RESULTS
    )

    print()
    print(
        "Original results file: UNCHANGED"
    )

    print("=" * 80)


def await_evaluate(
    dataset,
    metric
):
    from ragas import evaluate

    return evaluate(
        dataset=dataset,
        metrics=[metric],
        raise_exceptions=True,
        show_progress=False
    )


if __name__ == "__main__":
    main()