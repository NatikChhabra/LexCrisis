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


tasks = _load("lexcrisis_env.tasks", "lexcrisis_env/tasks.py")
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


def test_ground_truth_cites_law_that_is_in_force():
    """No scenario may cite the repealed Indian Evidence Act.

    The IEA 1872 was repealed by the Bharatiya Sakshya Adhiniyam 2023 with
    effect from 1 July 2024. Ground truth cites the BSA. Before this was fixed,
    the doctrine column marked an agent wrong for citing the statute actually
    in force. See issue #19.
    """
    cited = {
        entry.get("doctrine", "")
        for entry in tasks.PRIVILEGE_GROUND_TRUTH.values()
    }
    stale = {text for text in cited if "iea" in text.lower()}
    assert not stale, f"ground truth still cites the repealed IEA: {sorted(stale)}"


def test_repealed_citation_still_earns_credit():
    """A model citing the old IEA numbering is out of date, not wrong.

    Both forms must score the same, or the benchmark simply moves the unfairness
    from new models to old ones.
    """
    current = graders._doctrine_credit("BSA Section 132", "BSA Section 132")
    repealed = graders._doctrine_credit("IEA Section 126", "BSA Section 132")
    wrong = graders._doctrine_credit("BSA Section 999", "BSA Section 132")

    assert abs(current - repealed) < 1e-9, (
        f"BSA scored {current} but the equivalent IEA citation scored {repealed}."
    )
    assert wrong < 0.5, f"an unrelated section scored {wrong}; it must not."


def test_expert_keywords_do_not_hardcode_a_repealed_section():
    """The grader's own keyword list cited IEA s.45, not just the scenarios.

    Renumbering tasks.py alone dropped task_3 expert_score from 0.999 to 0.8333,
    because "section 45" was hardcoded in breakdown_task_3's expected keywords.
    Both numberings must now score alike.
    """
    base = "Special skill in toxicology and regulatory science, with relevant expertise under "
    findings_new = {"expert_assessed": {"qualification": base + "BSA Section 39."}}
    findings_old = {"expert_assessed": {"qualification": base + "IEA Section 45."}}

    new_score = graders.breakdown_task_3(findings_new, tasks.CRISIS_GROUND_TRUTH)["expert_score"]
    old_score = graders.breakdown_task_3(findings_old, tasks.CRISIS_GROUND_TRUTH)["expert_score"]

    assert new_score > 0.99, f"current citation scored only {new_score}"
    assert abs(new_score - old_score) < 1e-9, (
        f"BSA scored {new_score} but IEA scored {old_score}."
    )


# --------------------------------------------------------------- issue #25
# The graders take a ground_truth argument, so they must score the scenario
# they are handed and not the one shipped in tasks.py. Three places read the
# built-in scenario instead, which made held-out evaluation impossible.

_HELD_OUT_PRIVILEGE = {
    "XDOC-1": {"classification": "attorney_client", "doctrine": "BSA Section 132",
               "action": "withhold", "exception": "none"},
    "XDOC-2": {"classification": "attorney_client", "doctrine": "BSA Section 132",
               "action": "produce", "exception": "crime_fraud"},
}


def _held_out_answers():
    return {
        "privilege_classifications": {
            doc_id: {"classification": truth["classification"], "doctrine": truth["doctrine"]}
            for doc_id, truth in _HELD_OUT_PRIVILEGE.items()
        },
        "recommendations": {
            doc_id: {"action": truth["action"]}
            for doc_id, truth in _HELD_OUT_PRIVILEGE.items()
        },
        "exceptions_identified": [{"doc_id": "XDOC-2", "exception_type": "crime_fraud"}],
        "waivers_identified": [{"doc_id": "XDOC-2"}],
    }


def test_waivers_are_scored_against_the_scenario_supplied():
    """A correct waiver call on an unseen scenario must earn credit."""

    components = graders.breakdown_task_2(_held_out_answers(), _HELD_OUT_PRIVILEGE)
    assert components["waiver_f1"] > 0.99, (
        f"correct held-out waiver scored {components['waiver_f1']}; the grader is "
        "still reading WAIVER_EVENTS from the built-in scenario"
    )


def test_reciting_the_built_in_waivers_earns_nothing_elsewhere():
    """Naming documents that do not exist in the scenario must not score."""

    answers = _held_out_answers()
    answers["waivers_identified"] = [{"doc_id": d} for d in tasks.WAIVER_EVENTS]
    components = graders.breakdown_task_2(answers, _HELD_OUT_PRIVILEGE)
    assert components["waiver_f1"] < 0.01, (
        f"memorising the built-in scenario scored {components['waiver_f1']} on data "
        "that contains none of those documents"
    )


def test_built_in_waiver_scoring_is_unchanged():
    """The rewrite must be a no-op on the shipped scenario."""

    answers = {
        "privilege_classifications": {},
        "recommendations": {},
        "exceptions_identified": [],
        "waivers_identified": [{"doc_id": d} for d in tasks.WAIVER_EVENTS],
    }
    components = graders.breakdown_task_2(answers, tasks.PRIVILEGE_GROUND_TRUTH)
    assert components["waiver_f1"] > 0.99, components["waiver_f1"]


def test_discovery_objection_credits_the_law_in_force():
    """BSA numbering must score at least as well as the repealed IEA numbering."""

    def score(objections):
        findings = {"discovery_response": {"response_type": "privilege_log",
                                           "objections": objections}}
        return graders.breakdown_task_3(findings, tasks.CRISIS_GROUND_TRUTH)["discovery_score"]

    in_force = score("Withheld under BSA Section 132 and Section 134")
    repealed = score("Withheld under Section 126 of the Indian Evidence Act")
    assert in_force > 0.99, f"current law scored only {in_force}"
    assert in_force >= repealed, (
        f"repealed law scored {repealed} but law in force scored {in_force}"
    )


def test_ethical_resolution_bonus_is_not_welded_to_one_event_id():
    """A contributed scenario's own ethical event must be able to earn the bonus."""

    ground_truth = {
        "deadlines": {},
        "ethical_issues": {"XEVENT-9"},
        "adversarial_items": set(),
        "priority_order": ["XEVENT-9"],
    }
    findings = {
        "ethical_issues_flagged": [{
            "event_id": "XEVENT-9",
            "resolution": "Withdraw, screen the team, disclose to the former client "
                          "and obtain consent under rule 33.",
        }],
        "actions_taken": [{"event_id": "XEVENT-9"}],
    }
    components = graders.breakdown_task_3(findings, ground_truth)
    assert components["ethical_score"] > 0.9, (
        f"a well-written contributed scenario scored {components['ethical_score']}; "
        "the bonus is still hardcoded to EVENT-004"
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
