"""Check a contributed scenario, then prove the graders can actually score it.

Every number LexCrisis reports comes from scenarios written by the same person
who wrote the graders. The repair, as Prof. Gijs van Dijck put it, is scenarios
the system has not seen, drafted by practitioners along with their answer keys.

This script is the intake for those. It validates the file against the schema
the graders expect and then runs the real grader twice: once on the answer key
itself, which must score at the ceiling, and once on a deliberately wrong
submission, which must not. A contribution that passes both is known to
discriminate, rather than merely being well-formed.

    python validate_contribution.py contrib/example_privilege.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from lexcrisis_env.graders import breakdown_task_2, grade_task_2

CLASSIFICATIONS = {"attorney_client", "work_product", "both", "none", "waived"}
ACTIONS = {"withhold", "produce"}
EXCEPTIONS = {"none", "crime_fraud", "at_issue"}
REQUIRED_DOC_FIELDS = ("doc_id", "title", "content", "classification",
                       "doctrine", "action", "exception")

CEILING = 0.999


def _fail(problems: list[str], message: str) -> None:
    problems.append(message)


def validate(payload: dict) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for a contributed privilege scenario."""

    errors: list[str] = []
    warnings: list[str] = []

    if payload.get("task") != "task_2":
        _fail(errors, 'task must be "task_2"; only privilege review accepts '
                      "contributions so far")

    documents = payload.get("documents")
    if not isinstance(documents, list) or not documents:
        _fail(errors, "documents must be a non-empty list")
        return errors, warnings

    if len(documents) < 3:
        plural = "document" if len(documents) == 1 else "documents"
        warnings.append(f"only {len(documents)} {plural}; a scenario with fewer "
                        "than three rarely separates a good agent from a lucky one")

    seen: set[str] = set()
    for index, doc in enumerate(documents):
        where = f"documents[{index}]"
        for field in REQUIRED_DOC_FIELDS:
            if field not in doc:
                _fail(errors, f"{where} is missing {field}")
        doc_id = doc.get("doc_id", "")
        if doc_id in seen:
            _fail(errors, f"{where} repeats doc_id {doc_id!r}")
        seen.add(doc_id)

        if doc.get("classification") not in CLASSIFICATIONS:
            _fail(errors, f"{where} classification {doc.get('classification')!r} "
                          f"is not one of {sorted(CLASSIFICATIONS)}")
        if doc.get("action") not in ACTIONS:
            _fail(errors, f"{where} action {doc.get('action')!r} is not one of "
                          f"{sorted(ACTIONS)}")
        if doc.get("exception") not in EXCEPTIONS:
            _fail(errors, f"{where} exception {doc.get('exception')!r} is not one "
                          f"of {sorted(EXCEPTIONS)}")
        if not str(doc.get("content", "")).strip():
            _fail(errors, f"{where} has empty content; the agent sees this text "
                          "and nothing else")

        doctrine = str(doc.get("doctrine", "")).strip().lower()
        if doc.get("classification") == "none" and doctrine not in ("none", ""):
            warnings.append(f"{where} is classified none but cites {doc['doctrine']!r}")
        if "indian evidence act" in doctrine or "iea" in doctrine.split():
            warnings.append(
                f"{where} cites the Indian Evidence Act, repealed on 1 July 2024. "
                "The Bharatiya Sakshya Adhiniyam is the law in force; the repealed "
                "numbering still earns credit, so this is a note, not an error."
            )

    if not any(doc.get("exception") != "none" for doc in documents):
        warnings.append("no document carries an exception, so the waiver and "
                        "exception columns cannot be exercised at all")
    if len({doc.get("action") for doc in documents}) < 2:
        warnings.append("every document has the same produce/withhold call, so an "
                        "agent that always answers the same way scores full marks")

    return errors, warnings


def to_ground_truth(payload: dict) -> dict:
    return {
        doc["doc_id"]: {
            "classification": doc["classification"],
            "doctrine": doc["doctrine"],
            "action": doc["action"],
            "exception": doc["exception"],
        }
        for doc in payload["documents"]
    }


def _answers_from(ground_truth: dict) -> dict:
    return {
        "privilege_classifications": {
            doc_id: {"classification": truth["classification"],
                     "doctrine": truth["doctrine"]}
            for doc_id, truth in ground_truth.items()
        },
        "recommendations": {doc_id: {"action": truth["action"]}
                            for doc_id, truth in ground_truth.items()},
        "exceptions_identified": [
            {"doc_id": doc_id, "exception_type": truth["exception"]}
            for doc_id, truth in ground_truth.items()
            if truth["exception"] != "none"
        ],
        "waivers_identified": [
            {"doc_id": doc_id} for doc_id, truth in ground_truth.items()
            if truth["exception"] != "none"
        ],
    }


def _inverted(ground_truth: dict) -> dict:
    """A submission that withholds everything and classifies nothing."""

    return {
        "privilege_classifications": {},
        "recommendations": {doc_id: {"action": "withhold"} for doc_id in ground_truth},
        "exceptions_identified": [],
        "waivers_identified": [],
    }


def main(path: str) -> int:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    errors, warnings = validate(payload)

    for warning in warnings:
        print(f"note   {warning}")
    for error in errors:
        print(f"ERROR  {error}")
    if errors:
        print(f"\n{len(errors)} error(s). Not gradeable yet.")
        return 1

    ground_truth = to_ground_truth(payload)
    perfect = grade_task_2(_answers_from(ground_truth), ground_truth)
    lazy = grade_task_2(_inverted(ground_truth), ground_truth)

    print(f"\n{len(ground_truth)} documents parsed and graded.")
    print(f"  answer key scored against itself : {perfect}")
    print(f"  withhold-everything submission   : {lazy}")
    for name, value in breakdown_task_2(_answers_from(ground_truth), ground_truth).items():
        print(f"    {name:26} {value}")

    problems = 0
    if perfect < CEILING:
        print("\nERROR  the answer key does not score at the ceiling against itself, "
              "so something in it contradicts the grader")
        problems += 1
    if lazy >= perfect:
        print("\nERROR  a submission that withholds everything scores as well as the "
              "answer key; this scenario does not discriminate")
        problems += 1
    if problems:
        return 1

    print("\nGradeable, and it separates a correct answer from a lazy one.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
