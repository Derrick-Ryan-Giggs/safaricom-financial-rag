"""
evaluation/sql_eval.py

Runs the actual SQL path (retrieval.sql_query.run_query) against every
question in a ground-truth file (see evaluation/sql_ground_truth.py) and
checks whether the correct value comes back. This is the SQL-path
equivalent of evaluation/answer_quality.py's RAG evaluation -- the harness
explicitly flagged as missing in the project handoff ("had received no
formal evaluation before this session -- only spot-checked manually").

Verdicts:
    CORRECT     -- the expected value appears somewhere in the returned rows
    WRONG_VALUE -- SQL ran and returned rows, but not the expected value
    EMPTY       -- SQL ran but returned zero rows
    SQL_ERROR   -- SQL generation/execution failed (e.g. hallucinated
                   column, invalid GROUP BY) -- see retrieval.sql_query's
                   SQLGenerationError

Resumable: if --output already exists, questions already present in it are
skipped, matching the pattern used by evaluation/answer_quality.py.

Usage:
    uv run python -m evaluation.sql_eval --ground-truth evaluation/sql_ground_truth_v1.jsonl --output evaluation/sql_eval_v1.jsonl
"""

import argparse
import json

from retrieval.sql_query import SQLGenerationError, run_query

TOLERANCE = 0.01  # relative tolerance for float/Decimal comparison


def values_match(expected: str, rows: list[dict]) -> bool:
    """
    Check whether `expected` shows up anywhere in the returned rows, in any
    column -- not just the column the ground truth was generated from. The
    SQL generator might alias a column or select extra ones, so this checks
    "did the right number come back at all" rather than requiring an exact
    column-name match.
    """
    try:
        expected_num = float(expected)
    except ValueError:
        return False

    for row in rows:
        for value in row.values():
            try:
                if abs(float(value) - expected_num) <= TOLERANCE * max(abs(expected_num), 1):
                    return True
            except (TypeError, ValueError):
                continue
    return False


def load_processed_questions(output_path: str) -> set:
    processed = set()
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    processed.add(json.loads(line)["question"])
    except FileNotFoundError:
        pass
    return processed


def run_evaluation(ground_truth: list[dict], output_path: str) -> None:
    already_done = load_processed_questions(output_path)
    remaining = [g for g in ground_truth if g["question"] not in already_done]

    if already_done:
        print(f"Resuming: {len(already_done)} questions already processed, {len(remaining)} remaining.")

    verdict_counts = {"CORRECT": 0, "WRONG_VALUE": 0, "EMPTY": 0, "SQL_ERROR": 0}

    with open(output_path, "a", encoding="utf-8") as outfile:
        for i, gt in enumerate(remaining):
            generated_sql = None
            rows = None
            try:
                rows, generated_sql = run_query(gt["question"], return_sql=True)
                if not rows:
                    verdict = "EMPTY"
                elif values_match(gt["expected_value"], rows):
                    verdict = "CORRECT"
                else:
                    verdict = "WRONG_VALUE"
            except SQLGenerationError as e:
                generated_sql = e.sql
                verdict = "SQL_ERROR"

            verdict_counts[verdict] += 1

            record = {
                "question": gt["question"],
                "table": gt["table"],
                "column": gt["column"],
                "fiscal_year": gt["fiscal_year"],
                "expected_value": gt["expected_value"],
                "generated_sql": generated_sql,
                "returned_rows": rows,
                "verdict": verdict,
            }
            outfile.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            outfile.flush()

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(remaining)} remaining questions...")

    print(f"Done. Verdicts this run: {verdict_counts}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the SQL retrieval path against generated ground truth.")
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ground_truth = []
    with open(args.ground_truth, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ground_truth.append(json.loads(line))

    print(f"Loaded {len(ground_truth)} SQL ground-truth questions.")
    run_evaluation(ground_truth, args.output)


if __name__ == "__main__":
    main()