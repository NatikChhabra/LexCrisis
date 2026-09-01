"""Invariants every LexCrisis grader must satisfy.

These encode properties that were violated before the grader fixes:

  - an empty submission must score at the floor, not above it
  - engaging and being wrong must beat staying silent
  - a constant keyword-stuffed string must not earn substantial credit
  - partial coverage must not score the same as full correct coverage

Run with:  python tests/test_grader_invariants.py

(The repository root carries an __init__.py, which makes pytest treat the whole
tree as one package and fail to collect this file. Running it directly avoids
that; the functions are still named so pytest can pick them up if the packaging
is ever cleaned up.)
"""

import sys
import types
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Load the grader without importing lexcrisis_env/__init__.py, which pulls in
# openenv and a running server.
_pkg = types.ModuleType("lexcrisis_env")
_pkg.__path__ = [str(ROOT / "lexcrisis_env")]
sys.modules.setdefault("lexcrisis_env", _pkg)


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load("lexcrisis_env.tasks", "lexcrisis_env/tasks.py")
graders = _load("lexcrisis_env.graders", "lexcrisis_env/graders.py")

FLOOR = 0.001
TOLERANCE = 0.02


def test_empty_submission_scores_at_the_floor():
    """An agent that submits nothing must not be credited on any task."""
    for task_id, grade in graders.GRADERS.items():
        score = grade({}, graders.GROUND_TRUTH[task_id])
        assert score <= FLOOR + TOLERANCE, (
            f"{task_id}: empty submission scored {score}, expected ~{FLOOR}. "
            "Silence is being credited somewhere in the breakdown."
        )


def test_attempting_beats_silence_on_task_2():
    """A wrong but genuine answer must outscore submitting nothing."""
    ground_truth = graders.GROUND_TRUTH["task_2"]
    documents = list(ground_truth)

    silent = graders.grade_task_2({}, ground_truth)
    attempted = graders.grade_task_2(
        {
            "privilege_classifications": {
                doc_id: {"classification": "not_privileged", "doctrine": "none"}
                for doc_id in documents
            }
        },
        ground_truth,
    )
    assert attempted > silent, (
        f"silence scored {silent} and an attempt scored {attempted}. "
        "The reward gradient points at emitting nothing."
    )


def test_keyword_stuffing_earns_little():
    """A constant string of legal vocabulary must not carry the doctrine column."""
    ground_truth = graders.GROUND_TRUTH["task_2"]
    documents = list(ground_truth)
    stuffed = "section 126 129 iea crime-fraud at-issue"

    components = graders.breakdown_task_2(
        {
            "privilege_classifications": {
                doc_id: {"classification": "both", "doctrine": stuffed}
                for doc_id in documents
            }
        },
        ground_truth,
    )
    assert components["doctrine_accuracy"] < 0.35, (
        f"keyword stuffing reached doctrine_accuracy "
        f"{components['doctrine_accuracy']}, which is reward hacking, not analysis."
    )


def test_partial_coverage_does_not_match_full_coverage():
    """Handling 2 of 6 events must not score the same as handling all six."""
    ground_truth = graders.GROUND_TRUTH["task_3"]
    priority_order = list(ground_truth.get("priority_order", []))
    assert len(priority_order) >= 3, "task_3 ground truth has too few events to test"

    def ordering_score(events):
        findings = {"actions_taken": [{"event_id": event} for event in events]}
        return graders.breakdown_task_3(findings, ground_truth)["ordering_score"]

    full = ordering_score(priority_order)
    partial = ordering_score(priority_order[:2])
    assert partial < full, (
        f"two-of-{len(priority_order)} scored {partial} against {full} for full "
        "coverage. The ordering column pays for doing less."
    )


def test_final_score_matches_its_own_breakdown():
    """grade_task_2 must equal the documented weighted sum of its components."""
    weights = {
        "classification_accuracy": 0.35,
        "doctrine_accuracy": 0.20,
        "waiver_f1": 0.20,
        "exception_accuracy": 0.10,
        "recommendation_accuracy": 0.15,
    }
    ground_truth = graders.GROUND_TRUTH["task_2"]
    documents = list(ground_truth)

    findings = {
        "privilege_classifications": {
            doc_id: {"classification": "attorney_client", "doctrine": "none"}
            for doc_id in documents
        }
    }
    components = graders.breakdown_task_2(findings, ground_truth)
    recomputed = sum(weight * components[key] for key, weight in weights.items())
    reported = graders.grade_task_2(findings, ground_truth)

    assert abs(recomputed - reported) < 0.005, (
        f"grade_task_2 reported {reported} but its own breakdown implies "
        f"{round(recomputed, 4)}."
    )


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}\n      {exc}")
    print()
    print(f"{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
