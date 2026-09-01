"""Deterministic graders for the LexCrisis benchmark."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Set

from lexcrisis_env.tasks import (
    CONFLICT_DECISIONS,
    CONFLICT_RULES,
    CRISIS_GROUND_TRUTH,
    PRIVILEGE_GROUND_TRUTH,
    WAIVER_EVENTS,
    normalize,
)


# Phase-2 requires every task score to be strictly inside (0, 1).
_SCORE_FLOOR = 0.001
_SCORE_CEIL  = 0.999


def _clamp_score(raw: float) -> float:
    """Clamp a grader score to the open interval (0, 1)."""
    return round(max(_SCORE_FLOOR, min(raw, _SCORE_CEIL)), 4)


def _safe_divide(numerator: float, denominator: float) -> float:
    raw = numerator / denominator if denominator else 0.0
    return _clamp_score(raw)


def _f1(predicted: Set[str], actual: Set[str]) -> float:
    if not predicted and not actual:
        return _SCORE_CEIL
    if not predicted or not actual:
        return _SCORE_FLOOR
    true_positive = len(predicted & actual)
    precision = true_positive / len(predicted) if len(predicted) else 0.0
    recall = true_positive / len(actual) if len(actual) else 0.0
    if precision + recall == 0:
        return _SCORE_FLOOR
    raw = (2 * precision * recall) / (precision + recall)
    return _clamp_score(raw)


def _keyword_overlap(predicted: str, expected_keywords: Iterable[str]) -> float:
    keywords = {normalize(keyword) for keyword in expected_keywords if normalize(keyword)}
    if not keywords:
        return _SCORE_CEIL
    text = normalize(predicted)
    matched = sum(1 for keyword in keywords if keyword in text)
    raw = matched / len(keywords)
    return _clamp_score(raw)


def breakdown_task_1(findings: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, float]:
    """Return component scores for conflict-safe client intake."""

    predicted_pairs = {
        frozenset((entry.get("client_a", ""), entry.get("client_b", "")))
        for entry in findings.get("conflicts_identified", [])
        if entry.get("client_a") and entry.get("client_b")
    }
    actual_pairs = set(ground_truth.get("conflict_pairs", set()))
    pair_f1 = _f1({",".join(sorted(pair)) for pair in predicted_pairs}, {",".join(sorted(pair)) for pair in actual_pairs})

    predicted_decisions = findings.get("decisions", {})
    actual_decisions = ground_truth.get("correct_decisions", {})
    decision_hits = sum(
        1
        for client_id, expected in actual_decisions.items()
        if normalize(predicted_decisions.get(client_id)) == normalize(expected)
    )
    decision_accuracy = _safe_divide(decision_hits, len(actual_decisions))

    actual_rules = ground_truth.get("conflict_rules", {})
    correct_rules = 0.0
    for citation in findings.get("rule_citations", []):
        pair = frozenset((citation.get("client_a", ""), citation.get("client_b", "")))
        if pair not in actual_rules:
            continue
        predicted_rule = normalize(citation.get("rule", ""))
        expected_rule = normalize(actual_rules[pair])
        if predicted_rule == expected_rule:
            correct_rules += 1.0
        elif expected_rule and predicted_rule and (
            predicted_rule in expected_rule or expected_rule in predicted_rule
        ):
            correct_rules += 0.6
    rule_accuracy = _safe_divide(correct_rules, len(actual_rules))

    return {
        "pair_f1": pair_f1,
        "decision_accuracy": decision_accuracy,
        "rule_accuracy": min(rule_accuracy, _SCORE_CEIL),
    }


def grade_task_1(findings: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """Grade conflict-safe client intake."""

    components = breakdown_task_1(findings, ground_truth)
    score = (
        (0.45 * components["pair_f1"])
        + (0.35 * components["decision_accuracy"])
        + (0.20 * components["rule_accuracy"])
    )
    return _clamp_score(score)


def breakdown_task_2(findings: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, float]:
    """Return component scores for privilege review."""

    classifications = findings.get("privilege_classifications", {})
    recommendations = findings.get("recommendations", {})
    exceptions = findings.get("exceptions_identified", [])
    waivers = findings.get("waivers_identified", [])

    classification_score = 0.0
    doctrine_score = 0.0
    recommendation_score = 0.0

    for doc_id, truth in ground_truth.items():
        predicted = classifications.get(doc_id, {})
        predicted_class = normalize(predicted.get("classification"))
        truth_class = normalize(truth.get("classification"))
        if predicted_class == truth_class:
            classification_score += 1.0
        elif predicted_class in {"attorney_client", "work_product", "both"} and truth_class in {
            "attorney_client",
            "work_product",
            "both",
        }:
            classification_score += 0.5

        predicted_doctrine = normalize(predicted.get("doctrine"))
        truth_doctrine = normalize(truth.get("doctrine"))
        if truth_doctrine:
            # Score against this document's doctrine only. The previous version
            # also accepted the generic terms "iea", "section", "126", "129",
            # "crime-fraud" and "at-issue", so one constant string containing
            # them scored 0.81 here regardless of the document.
            doctrine_score += _keyword_overlap(predicted_doctrine, [truth_doctrine])
        elif predicted_doctrine == "none":
            # An explicit assertion that no doctrine applies. Correct.
            doctrine_score += 1.0
        elif predicted_doctrine:
            # Asserted some doctrine where none applies. Wrong, but engaged.
            doctrine_score += 0.3
        # Field absent entirely: no credit. Previously this scored a full 1.0,
        # which gave an empty submission 0.0609 on this task.

        predicted_action = normalize(recommendations.get(doc_id, {}).get("action"))
        truth_action = normalize(truth.get("action"))
        if predicted_action == truth_action:
            recommendation_score += 1.0

    document_count = len(ground_truth)
    classification_accuracy = _safe_divide(classification_score, document_count)
    doctrine_accuracy = _safe_divide(doctrine_score, document_count)
    recommendation_accuracy = _safe_divide(recommendation_score, document_count)

    predicted_waivers = {entry.get("doc_id", "") for entry in waivers if entry.get("doc_id")}
    actual_waivers = set(WAIVER_EVENTS.keys())
    waiver_f1 = _f1(predicted_waivers, actual_waivers)

    exception_lookup = {
        entry.get("doc_id", ""): normalize(entry.get("exception_type"))
        for entry in exceptions
        if entry.get("doc_id")
    }
    exception_hits = 0.0
    exception_targets = 0
    for doc_id, truth in ground_truth.items():
        expected_exception = normalize(truth.get("exception"))
        if expected_exception == "none":
            continue
        exception_targets += 1
        if exception_lookup.get(doc_id) == expected_exception:
            exception_hits += 1.0
    exception_accuracy = _safe_divide(exception_hits, exception_targets)

    return {
        "classification_accuracy": classification_accuracy,
        "doctrine_accuracy": doctrine_accuracy,
        "waiver_f1": waiver_f1,
        "exception_accuracy": exception_accuracy,
        "recommendation_accuracy": recommendation_accuracy,
    }


def grade_task_2(findings: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """Grade privilege review."""

    components = breakdown_task_2(findings, ground_truth)
    score = (
        (0.35 * components["classification_accuracy"])
        + (0.20 * components["doctrine_accuracy"])
        + (0.20 * components["waiver_f1"])
        + (0.10 * components["exception_accuracy"])
        + (0.15 * components["recommendation_accuracy"])
    )
    return _clamp_score(score)


def breakdown_task_3(findings: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, float]:
    """Return component scores for litigation incident command."""

    deadlines = ground_truth.get("deadlines", {})
    recorded_deadlines = findings.get("deadlines_met", {})
    deadline_score = 0.0
    for event_id, truth in deadlines.items():
        deadline_step = truth.get("deadline_step", 10**6)
        if event_id not in recorded_deadlines:
            continue
        met_step = recorded_deadlines[event_id].get("step", 10**6)
        if met_step <= deadline_step:
            deadline_score += 1.0
        else:
            deadline_score += 0.25
    deadline_accuracy = _safe_divide(deadline_score, len(deadlines))

    ethical_findings = findings.get("ethical_issues_flagged", [])
    ethical_events = {entry.get("event_id", "") for entry in ethical_findings if entry.get("event_id")}
    ethical_f1 = _f1(ethical_events, set(ground_truth.get("ethical_issues", set())))
    ethical_resolution_bonus = 0.0
    for entry in ethical_findings:
        if entry.get("event_id") != "EVENT-004":
            continue
        ethical_resolution_bonus = _keyword_overlap(
            entry.get("resolution", ""),
            ["withdraw", "screen", "consent", "disclose", "rule 33", "former client"],
        )
        break

    adversarial_findings = findings.get("adversarial_flagged", [])
    adversarial_events = {entry.get("item_id", "") for entry in adversarial_findings if entry.get("item_id")}
    adversarial_f1 = _f1(adversarial_events, set(ground_truth.get("adversarial_items", set())))

    discovery = findings.get("discovery_response", {})
    discovery_response_score = 0.0
    if discovery:
        response_type = normalize(discovery.get("response_type"))
        objections = normalize(discovery.get("objections"))
        if response_type in {"privilege_log", "object", "partial_produce"}:
            discovery_response_score += 0.6
        if any(term in objections for term in ("privilege", "section 126", "section 129", "advocate")):
            discovery_response_score += 0.4
    discovery_response_score = min(discovery_response_score, _SCORE_CEIL)

    expert_assessment = findings.get("expert_assessed", {})
    expert_score = 0.0
    if expert_assessment:
        expert_score = _keyword_overlap(
            expert_assessment.get("qualification", ""),
            ["special skill", "science", "regulatory", "toxicology", "section 45", "relevant"],
        )

    action_order = []
    for action in findings.get("actions_taken", []):
        event_id = action.get("event_id")
        if event_id and event_id not in action_order:
            action_order.append(event_id)
    # Score against the FULL ground-truth ordering. Filtering it down to events
    # the agent happened to touch meant an agent that handled 2 of 6 events in
    # the right order scored 0.999, identical to handling all six, because the
    # untouched events could not contribute an incorrectly-ordered pair.
    full_order = list(ground_truth.get("priority_order", []))
    ordering_pairs = 0
    correct_pairs = 0
    for left_index, left_event in enumerate(full_order):
        for right_event in full_order[left_index + 1 :]:
            ordering_pairs += 1
            if (
                left_event in action_order
                and right_event in action_order
                and action_order.index(left_event) < action_order.index(right_event)
            ):
                correct_pairs += 1
    ordering_score = _safe_divide(correct_pairs, ordering_pairs) if ordering_pairs else _SCORE_FLOOR

    return {
        "deadline_accuracy": deadline_accuracy,
        "ethical_score": (0.7 * ethical_f1) + (0.3 * ethical_resolution_bonus),
        "adversarial_f1": adversarial_f1,
        "discovery_score": discovery_response_score,
        "expert_score": expert_score,
        "ordering_score": ordering_score,
    }


def grade_task_3(findings: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """Grade litigation incident command."""

    components = breakdown_task_3(findings, ground_truth)
    score = (
        (0.30 * components["deadline_accuracy"])
        + (0.20 * components["ethical_score"])
        + (0.15 * components["adversarial_f1"])
        + (0.15 * components["discovery_score"])
        + (0.10 * components["expert_score"])
        + (0.10 * components["ordering_score"])
    )
    return _clamp_score(score)


GRADERS = {
    "task_1": grade_task_1,
    "task_2": grade_task_2,
    "task_3": grade_task_3,
}


GROUND_TRUTH = {
    "task_1": {
        "conflict_pairs": set(CONFLICT_RULES.keys()),
        "conflict_rules": CONFLICT_RULES,
        "correct_decisions": dict(CONFLICT_DECISIONS),
    },
    "task_2": PRIVILEGE_GROUND_TRUTH,
    "task_3": CRISIS_GROUND_TRUTH,
}


GRADER_BREAKDOWNS = {
    "task_1": breakdown_task_1,
    "task_2": breakdown_task_2,
    "task_3": breakdown_task_3,
}
