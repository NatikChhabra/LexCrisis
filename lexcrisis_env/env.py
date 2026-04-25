"""Core environment logic for the LexCrisis benchmark."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

from openenv.core.env_server import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from lexcrisis_env.graders import GROUND_TRUTH, GRADERS, _SCORE_CEIL, _SCORE_FLOOR
from lexcrisis_env.models import (
    Action,
    DeadlineSummary,
    DocumentSummary,
    EnvironmentState,
    Observation,
    Reward,
)
from lexcrisis_env.tasks import (
    CLIENTS,
    CONFLICT_RULES,
    CRISIS_EVENTS,
    CRISIS_GROUND_TRUTH,
    PRIVILEGE_DOCUMENTS,
    PRIVILEGE_GROUND_TRUTH,
    TASK_ACTIONS,
    TASK_DEFINITIONS,
    TERMINAL_ACTIONS,
    WAIVER_EVENTS,
    first_matching,
    normalize,
)

BENCHMARK_NAME = "lexcrisis"
BENCHMARK_VERSION = "1.0.0"
VERIFIER_COLUMNS = (
    "outcome_correctness",
    "process_compliance",
    "deadline_or_latency",
    "safety_or_anti_cheat",
)


@dataclass
class StepOutcome:
    """Internal action result before the reward delta is computed."""

    milestone_bonus: float = 0.0
    penalty: float = 0.0
    feedback: str = ""
    verifier_signals: Dict[str, float] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)


class LexCrisisEngine:
    """Stateful legal-ops engine for litigation incident response."""

    def __init__(self) -> None:
        self._episode_id = str(uuid4())
        self._task_id = "task_1"
        self._step_count = 0
        self._score = _SCORE_FLOOR
        self._done = False
        self._last_reward = 0.0
        self._last_reward_model = Reward(
            value=0.0,
            score_delta=0.0,
            milestone_bonus=0.0,
            penalty=0.0,
            verifier_signals=self._default_verifier_signals(),
            flags=[],
            reason="Environment initialized.",
        )
        self._feedback = "Environment ready."
        self._current_content: Optional[str] = None
        self._ethical_alerts: List[str] = []
        self._findings: Dict[str, Any] = {}
        self._action_history: List[str] = []
        self._cumulative_reward = 0.0
        self._seed: Optional[int] = None
        self._episode_config: Dict[str, Any] = {}
        self._signal_totals = self._default_verifier_signals()
        self._signal_steps = 0
        self._clients_for_episode = list(CLIENTS)
        self._documents_for_episode = list(PRIVILEGE_DOCUMENTS)
        self._events_for_episode = list(CRISIS_EVENTS)
        self._client_lookup = {client.client_id: client for client in CLIENTS}
        self._document_lookup = {document.doc_id: document for document in PRIVILEGE_DOCUMENTS}
        self._event_lookup = {event.event_id: event for event in CRISIS_EVENTS}
        self._deadline_truth = copy.deepcopy(CRISIS_GROUND_TRUTH["deadlines"])
        self.reset()

    def reset(
        self,
        task_id: str | None = None,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        episode_config: Optional[Dict[str, Any]] = None,
    ) -> Observation:
        """Reset the active episode and return a clean observation."""

        selected_task = task_id or self._task_id
        if selected_task not in TASK_DEFINITIONS:
            selected_task = "task_1"

        self._task_id = selected_task
        self._episode_id = episode_id or str(uuid4())
        self._seed = seed
        self._configure_episode(selected_task, seed=seed, episode_config=episode_config)
        self._step_count = 0
        self._score = _SCORE_FLOOR
        self._done = False
        self._last_reward = 0.0
        self._cumulative_reward = 0.0
        self._current_content = None
        self._ethical_alerts = []
        self._findings = self._empty_findings(selected_task)
        self._action_history = []
        self._signal_totals = self._default_verifier_signals()
        self._signal_steps = 0
        definition = TASK_DEFINITIONS[selected_task]
        self._feedback = (
            f"Environment reset for {definition.name}. "
            f"Work the problem like a legal operations team under pressure and submit before step {definition.max_steps}."
        )
        self._last_reward_model = Reward(
            value=0.0,
            score_delta=0.0,
            milestone_bonus=0.0,
            penalty=0.0,
            verifier_signals=self._default_verifier_signals(),
            flags=[],
            reason="Environment reset.",
        )
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Apply one action and return the next observation."""

        if self._done:
            self._feedback = "Episode already complete. Call reset() before taking more actions."
            observation = self._build_observation()
            return observation, 0.0, True, self._info_payload()

        self._step_count += 1
        self._current_content = None

        old_score = self._run_grader()
        outcome = self._dispatch(action)
        loop_penalty = self._loop_penalty(action)
        penalty = outcome.penalty + loop_penalty
        verifier_signals = dict(outcome.verifier_signals)
        flags = list(outcome.flags)

        if loop_penalty < 0:
            verifier_signals["process_compliance"] = min(verifier_signals["process_compliance"], 0.25)
            verifier_signals["safety_or_anti_cheat"] = min(verifier_signals["safety_or_anti_cheat"], 0.25)
            flags.append("loop_penalty")

        new_score = self._run_grader()
        score_delta = round(new_score - old_score, 4)
        reward_value = round(score_delta + outcome.milestone_bonus + penalty, 4)

        terminal_action = TERMINAL_ACTIONS[self._task_id]
        if action.action_type == terminal_action:
            self._done = True

        if self._step_count >= TASK_DEFINITIONS[self._task_id].max_steps:
            self._done = True
            if action.action_type != terminal_action:
                penalty -= 0.03
                reward_value = round(reward_value - 0.03, 4)
                verifier_signals["deadline_or_latency"] = 0.0
                verifier_signals["process_compliance"] = min(verifier_signals["process_compliance"], 0.0)
                verifier_signals["safety_or_anti_cheat"] = min(verifier_signals["safety_or_anti_cheat"], 0.0)
                flags.append("timeout_without_submit")
                outcome.feedback += " Max steps reached without submission."

        self._last_reward = reward_value
        self._cumulative_reward = round(self._cumulative_reward + reward_value, 4)
        self._score = round(max(_SCORE_FLOOR, min(new_score, _SCORE_CEIL)), 4)

        if self._done:
            outcome.feedback += f" Final score: {self._score:.2f}."

        self._feedback = outcome.feedback
        self._apply_step_signals(verifier_signals)
        self._last_reward_model = Reward(
            value=reward_value,
            score_delta=score_delta,
            milestone_bonus=round(outcome.milestone_bonus, 4),
            penalty=round(penalty, 4),
            verifier_signals={key: round(verifier_signals[key], 4) for key in VERIFIER_COLUMNS},
            flags=flags,
            reason=outcome.feedback,
        )

        observation = self._build_observation()
        self._action_history.append(self._fingerprint(action))
        return observation, reward_value, self._done, self._info_payload()

    def state(self) -> EnvironmentState:
        """Return the exact state contract requested in the prompt."""

        observation = self._build_observation()
        return EnvironmentState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            observation=observation.model_dump(exclude={"reward", "done", "metadata"}),
            reward=self._last_reward,
            done=self._done,
        )

    @property
    def last_score(self) -> float:
        return round(max(_SCORE_FLOOR, min(self._score, _SCORE_CEIL)), 4)

    @property
    def episode_id(self) -> str:
        return self._episode_id

    @property
    def episode_config(self) -> Dict[str, Any]:
        return copy.deepcopy(self._episode_config)

    def close(self) -> None:
        """No-op cleanup for API compatibility."""

    def episode_info(self) -> Dict[str, Any]:
        """Return UI-friendly episode metadata without changing the OpenEnv state contract."""

        payload = self._info_payload()
        payload.update(
            {
                "done": self._done,
                "mode": "simulation",
                "benchmark": BENCHMARK_NAME,
                "last_reward": self._last_reward,
                "last_reward_reason": self._last_reward_model.reason,
            }
        )
        return payload

    def _default_verifier_signals(self) -> Dict[str, float]:
        return {column: 0.0 for column in VERIFIER_COLUMNS}

    def _make_outcome(
        self,
        milestone_bonus: float = 0.0,
        penalty: float = 0.0,
        feedback: str = "",
        *,
        outcome_correctness: float = 1.0,
        process_compliance: float = 1.0,
        deadline_or_latency: float = 1.0,
        safety_or_anti_cheat: float = 1.0,
        flags: Optional[List[str]] = None,
    ) -> StepOutcome:
        return StepOutcome(
            milestone_bonus=milestone_bonus,
            penalty=penalty,
            feedback=feedback,
            verifier_signals={
                "outcome_correctness": max(0.0, min(outcome_correctness, 1.0)),
                "process_compliance": max(0.0, min(process_compliance, 1.0)),
                "deadline_or_latency": max(0.0, min(deadline_or_latency, 1.0)),
                "safety_or_anti_cheat": max(0.0, min(safety_or_anti_cheat, 1.0)),
            },
            flags=flags or [],
        )

    def _empty_findings(self, task_id: str) -> Dict[str, Any]:
        if task_id == "task_1":
            return {
                "reviewed_clients": [],
                "conflicts_identified": [],
                "rule_citations": [],
                "decisions": {},
            }
        if task_id == "task_2":
            return {
                "reviewed_documents": [],
                "privilege_classifications": {},
                "waivers_identified": [],
                "exceptions_identified": [],
                "recommendations": {},
            }
        return {
            "reviewed_events": [],
            "deadlines_met": {},
            "adversarial_flagged": [],
            "ethical_issues_flagged": [],
            "actions_taken": [],
            "expert_assessed": {},
            "discovery_response": {},
        }

    def _configure_episode(
        self,
        task_id: str,
        *,
        seed: Optional[int],
        episode_config: Optional[Dict[str, Any]],
    ) -> None:
        config = copy.deepcopy(episode_config or {})
        rng = random.Random(seed) if seed is not None else None

        style = str(config.get("content_style") or ("ops_brief" if rng else "default"))
        self._episode_config = {"seed": seed, "content_style": style}
        self._client_lookup = {client.client_id: client for client in CLIENTS}
        self._document_lookup = {document.doc_id: document for document in PRIVILEGE_DOCUMENTS}
        self._event_lookup = {event.event_id: event for event in CRISIS_EVENTS}
        self._clients_for_episode = self._ordered_items(
            CLIENTS,
            key=lambda item: item.client_id,
            explicit_order=config.get("client_order"),
            rng=rng,
        )
        self._documents_for_episode = self._ordered_items(
            PRIVILEGE_DOCUMENTS,
            key=lambda item: item.doc_id,
            explicit_order=config.get("document_order"),
            rng=rng,
        )
        self._events_for_episode = self._ordered_items(
            CRISIS_EVENTS,
            key=lambda item: item.event_id,
            explicit_order=config.get("event_order"),
            rng=rng,
        )
        self._deadline_truth = copy.deepcopy(CRISIS_GROUND_TRUTH["deadlines"])
        deadline_overrides = {
            event_id: details["deadline_step"]
            for event_id, details in self._deadline_truth.items()
        }
        requested_overrides = config.get("deadline_overrides")
        if isinstance(requested_overrides, dict):
            for event_id, deadline in requested_overrides.items():
                if event_id in self._deadline_truth and isinstance(deadline, int):
                    self._deadline_truth[event_id]["deadline_step"] = max(2, deadline)
        elif rng and task_id == "task_3":
            for event_id in deadline_overrides:
                base = self._deadline_truth[event_id]["deadline_step"]
                self._deadline_truth[event_id]["deadline_step"] = max(2, base + rng.randint(-2, 2))
        deadline_overrides = {
            event_id: details["deadline_step"]
            for event_id, details in self._deadline_truth.items()
        }
        self._episode_config.update(
            {
                "client_order": [client.client_id for client in self._clients_for_episode],
                "document_order": [document.doc_id for document in self._documents_for_episode],
                "event_order": [event.event_id for event in self._events_for_episode],
                "deadline_overrides": deadline_overrides,
            }
        )

    def _ordered_items(
        self,
        items: Sequence[Any],
        *,
        key: Any,
        explicit_order: Optional[Iterable[str]],
        rng: Optional[random.Random],
    ) -> List[Any]:
        ordered = list(items)
        if explicit_order:
            lookup = {key(item): item for item in ordered}
            picked = [lookup[item_id] for item_id in explicit_order if item_id in lookup]
            remaining = [item for item in ordered if key(item) not in set(explicit_order)]
            return picked + remaining
        if rng:
            rng.shuffle(ordered)
        return ordered

    def _styled_title(self, base_title: str) -> str:
        style = self._episode_config.get("content_style", "default")
        if style == "default":
            return base_title
        label = str(style).replace("_", " ").title()
        return f"{base_title} [{label}]"

    def _render_client_content(self, client_id: str) -> str:
        client = self._client_lookup[client_id]
        title = self._styled_title(client.name)
        return (
            f"{title} ({client.client_id})\n"
            f"Type: {client.client_type}\n"
            f"Summary: {client.summary}\n"
            f"Details: {client.details}\n"
            f"Relationships: {', '.join(client.relationships) if client.relationships else 'None'}"
        )

    def _render_document_content(self, doc_id: str) -> str:
        document = self._document_lookup[doc_id]
        title = self._styled_title(document.title)
        return (
            f"{title} ({document.doc_id})\n"
            f"Doctrine hint: {document.doctrine}\n"
            f"Content: {document.content}"
        )

    def _render_event_content(self, event_id: str) -> str:
        event = self._event_lookup[event_id]
        title = self._styled_title(event.title)
        return (
            f"{title} ({event.event_id})\n"
            f"Type: {event.event_type}\n"
            f"Deadline step: {self._event_deadline(event.event_id)}\n"
            f"Consequence: {event.consequence}\n"
            f"Scenario: {event.content}"
        )

    def _event_deadline(self, event_id: str) -> int:
        if event_id in self._deadline_truth:
            return int(self._deadline_truth[event_id]["deadline_step"])
        return int(self._event_lookup[event_id].deadline_step)

    def _run_grader(self) -> float:
        grader = GRADERS[self._task_id]
        truth = copy.deepcopy(GROUND_TRUTH[self._task_id])
        if self._task_id == "task_3":
            truth["deadlines"] = copy.deepcopy(self._deadline_truth)
        try:
            return grader(copy.deepcopy(self._findings), truth)
        except Exception:
            return _SCORE_FLOOR

    def _build_observation(self) -> Observation:
        definition = TASK_DEFINITIONS[self._task_id]
        documents: List[DocumentSummary]
        active_deadlines: List[DeadlineSummary] = []

        if self._task_id == "task_1":
            documents = [
                DocumentSummary(
                    item_id=client.client_id,
                    title=self._styled_title(client.name),
                    item_type="client_intake",
                    category=client.client_type,
                )
                for client in self._clients_for_episode
            ]
        elif self._task_id == "task_2":
            documents = [
                DocumentSummary(
                    item_id=document.doc_id,
                    title=self._styled_title(document.title),
                    item_type="litigation_document",
                    category="privilege_review",
                )
                for document in self._documents_for_episode
            ]
        else:
            documents = [
                DocumentSummary(
                    item_id=event.event_id,
                    title=self._styled_title(event.title),
                    item_type="crisis_event",
                    category=event.event_type,
                )
                for event in self._events_for_episode
            ]
            for event in self._events_for_episode:
                deadline = self._event_deadline(event.event_id)
                if deadline <= 0:
                    continue
                if event.event_id in self._findings.get("deadlines_met", {}):
                    continue
                remaining = deadline - self._step_count
                if remaining > 0:
                    active_deadlines.append(
                        DeadlineSummary(
                            item_id=event.event_id,
                            title=self._styled_title(event.title),
                            steps_remaining=remaining,
                            consequence=event.consequence,
                        )
                    )

        return Observation(
            task_id=definition.task_id,
            task_name=definition.name,
            difficulty=definition.difficulty,  # type: ignore[arg-type]
            task_description=definition.description,
            documents=documents,
            current_content=self._current_content,
            available_actions=TASK_ACTIONS[self._task_id],
            findings=copy.deepcopy(self._findings),
            feedback=self._feedback,
            step_count=self._step_count,
            max_steps=definition.max_steps,
            active_deadlines=active_deadlines,
            ethical_alerts=list(self._ethical_alerts),
            episode_config=copy.deepcopy(self._episode_config),
            done=self._done,
            reward=self._last_reward,
            metadata=self._info_payload(),
        )

    def _info_payload(self) -> Dict[str, Any]:
        averages = self._default_verifier_signals()
        if self._signal_steps:
            averages = {
                key: round(self._signal_totals[key] / self._signal_steps, 4)
                for key in VERIFIER_COLUMNS
            }
        return {
            "episode_id": self._episode_id,
            "task_id": self._task_id,
            "step_count": self._step_count,
            "score": round(max(_SCORE_FLOOR, min(self._score, _SCORE_CEIL)), 4),
            "cumulative_reward": self._cumulative_reward,
            "reward_breakdown": self._last_reward_model.model_dump(),
            "episode_config": copy.deepcopy(self._episode_config),
            "verifier_columns": {
                "last_step": self._last_reward_model.verifier_signals,
                "episode_average": averages,
            },
        }

    def _fingerprint(self, action: Action) -> str:
        return f"{action.action_type}|{normalize(action.parameters)}"

    def _loop_penalty(self, action: Action) -> float:
        fingerprint = self._fingerprint(action)
        if action.action_type == "noop":
            overdue_count = 0
            if self._task_id == "task_3":
                for event_id, details in self._deadline_truth.items():
                    if self._step_count > details["deadline_step"] and event_id not in self._findings["deadlines_met"]:
                        overdue_count += 1
            return -0.02 - (0.03 * overdue_count)
        recent = self._action_history[-2:]
        if fingerprint in recent:
            return -0.02
        return 0.0

    def _apply_step_signals(self, signals: Dict[str, float]) -> None:
        self._signal_steps += 1
        for key in VERIFIER_COLUMNS:
            self._signal_totals[key] += signals.get(key, 0.0)

    def _target_event_for_motion(self, motion_type: str) -> str:
        normalized = normalize(motion_type)
        if any(term in normalized for term in ("injunction", "tro", "opposition")):
            return "EVENT-002"
        if any(term in normalized for term in ("transfer", "consolidation", "forum")):
            return "EVENT-005"
        return ""

    def _require_reviewed_clients(self, client_ids: Iterable[str]) -> Optional[StepOutcome]:
        reviewed = set(self._findings["reviewed_clients"])
        missing = [client_id for client_id in client_ids if client_id not in reviewed]
        if not missing:
            return None
        missing_text = ", ".join(missing)
        return self._make_outcome(
            penalty=-0.03,
            feedback=f"Review prerequisite missing: inspect {missing_text} before taking this action.",
            outcome_correctness=0.0,
            process_compliance=0.0,
            deadline_or_latency=1.0,
            safety_or_anti_cheat=0.0,
            flags=["review_prerequisite_missing"],
        )

    def _require_reviewed_document(self, doc_id: str) -> Optional[StepOutcome]:
        if doc_id in self._findings["reviewed_documents"]:
            return None
        return self._make_outcome(
            penalty=-0.03,
            feedback=f"Review prerequisite missing: inspect {doc_id} before taking this action.",
            outcome_correctness=0.0,
            process_compliance=0.0,
            deadline_or_latency=1.0,
            safety_or_anti_cheat=0.0,
            flags=["review_prerequisite_missing"],
        )

    def _require_reviewed_event(self, event_id: str) -> Optional[StepOutcome]:
        if event_id in self._findings["reviewed_events"]:
            return None
        return self._make_outcome(
            penalty=-0.03,
            feedback=f"Review prerequisite missing: inspect {event_id} before taking this action.",
            outcome_correctness=0.0,
            process_compliance=0.0,
            deadline_or_latency=1.0,
            safety_or_anti_cheat=0.0,
            flags=["review_prerequisite_missing"],
        )

    def _dispatch(self, action: Action) -> StepOutcome:
        handlers = {
            "review_client": self._review_client,
            "check_conflict": self._check_conflict,
            "cite_rule": self._cite_rule,
            "accept_client": self._decide_client,
            "decline_client": self._decide_client,
            "submit_intake": self._submit,
            "review_document": self._review_document,
            "classify_privilege": self._classify_privilege,
            "identify_waiver": self._identify_waiver,
            "identify_exception": self._identify_exception,
            "recommend_action": self._recommend_action,
            "submit_review": self._submit,
            "review_event": self._review_event,
            "issue_litigation_hold": self._issue_litigation_hold,
            "file_motion": self._file_motion,
            "respond_discovery": self._respond_discovery,
            "assess_expert": self._assess_expert,
            "flag_adversarial": self._flag_adversarial,
            "flag_ethical_issue": self._flag_ethical_issue,
            "submit_triage": self._submit,
            "noop": self._noop,
        }
        handler = handlers.get(action.action_type)
        if handler is None:
            return self._make_outcome(
                penalty=-0.05,
                feedback=f"Unknown action '{action.action_type}'.",
                outcome_correctness=0.0,
                process_compliance=0.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=0.0,
                flags=["unknown_action"],
            )
        return handler(action)

    def _review_client(self, action: Action) -> StepOutcome:
        client_id = str(action.parameters.get("client_id", "")).upper()
        client = self._client_lookup.get(client_id)
        if client is None:
            return self._make_outcome(
                penalty=-0.03,
                feedback=f"Unknown client '{client_id}'.",
                outcome_correctness=0.0,
                process_compliance=0.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
                flags=["unknown_target"],
            )
        reviewed = self._findings["reviewed_clients"]
        milestone_bonus = 0.02 if client_id not in reviewed else 0.0
        penalty = -0.01 if client_id in reviewed else 0.0
        if client_id not in reviewed:
            reviewed.append(client_id)
        self._current_content = self._render_client_content(client_id)
        return self._make_outcome(
            milestone_bonus=milestone_bonus,
            penalty=penalty,
            feedback=f"Reviewed client intake for {client.name}.",
            outcome_correctness=1.0,
            process_compliance=1.0 if penalty == 0 else 0.5,
            deadline_or_latency=1.0,
            safety_or_anti_cheat=1.0,
        )

    def _check_conflict(self, action: Action) -> StepOutcome:
        client_a = str(action.parameters.get("client_a", "")).upper()
        client_b = str(action.parameters.get("client_b", "")).upper()
        if not client_a or not client_b or client_a == client_b:
            return self._make_outcome(
                penalty=-0.03,
                feedback="Provide two distinct client IDs to check a conflict.",
                outcome_correctness=0.0,
                process_compliance=0.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        prerequisite = self._require_reviewed_clients([client_a, client_b])
        if prerequisite is not None:
            return prerequisite
        pair = frozenset((client_a, client_b))
        existing = {
            frozenset((entry["client_a"], entry["client_b"]))
            for entry in self._findings["conflicts_identified"]
        }
        if pair not in existing:
            self._findings["conflicts_identified"].append({"client_a": client_a, "client_b": client_b})
        if pair in CONFLICT_RULES:
            bonus = 0.03 if pair not in existing else 0.0
            return self._make_outcome(
                milestone_bonus=bonus,
                feedback=f"Conflict identified between {client_a} and {client_b}.",
                outcome_correctness=1.0,
                process_compliance=1.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        return self._make_outcome(
            penalty=-0.04,
            feedback=f"No conflict exists between {client_a} and {client_b}.",
            outcome_correctness=0.0,
            process_compliance=1.0,
            deadline_or_latency=1.0,
            safety_or_anti_cheat=1.0,
        )

    def _cite_rule(self, action: Action) -> StepOutcome:
        client_a = str(action.parameters.get("client_a", "")).upper()
        client_b = str(action.parameters.get("client_b", "")).upper()
        rule = str(action.parameters.get("rule", ""))
        if not client_a or not client_b or not rule:
            return self._make_outcome(
                penalty=-0.03,
                feedback="Rule citation requires client_a, client_b, and rule.",
                outcome_correctness=0.0,
                process_compliance=0.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        prerequisite = self._require_reviewed_clients([client_a, client_b])
        if prerequisite is not None:
            return prerequisite
        pair = frozenset((client_a, client_b))
        citations = self._findings["rule_citations"]
        citations[:] = [entry for entry in citations if frozenset((entry["client_a"], entry["client_b"])) != pair]
        citations.append({"client_a": client_a, "client_b": client_b, "rule": rule})
        expected = normalize(CONFLICT_RULES.get(pair, ""))
        provided = normalize(rule)
        if expected and (provided == expected or provided in expected or expected in provided):
            return self._make_outcome(
                milestone_bonus=0.02,
                feedback=f"Applied the correct conflict rule for {client_a} and {client_b}.",
                outcome_correctness=1.0,
                process_compliance=1.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        return self._make_outcome(
            penalty=-0.03,
            feedback=f"Rule citation for {client_a} and {client_b} does not match the expected basis.",
            outcome_correctness=0.0,
            process_compliance=1.0,
            deadline_or_latency=1.0,
            safety_or_anti_cheat=1.0,
        )

    def _decide_client(self, action: Action) -> StepOutcome:
        client_id = str(action.parameters.get("client_id", "")).upper()
        if not client_id:
            return self._make_outcome(
                penalty=-0.03,
                feedback="Client decision requires a client_id.",
                outcome_correctness=0.0,
                process_compliance=0.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        prerequisite = self._require_reviewed_clients([client_id])
        if prerequisite is not None:
            return prerequisite
        decision = "accept" if action.action_type == "accept_client" else "decline"
        penalty = -0.01 if client_id in self._findings["decisions"] else 0.0
        self._findings["decisions"][client_id] = decision
        expected = GROUND_TRUTH["task_1"]["correct_decisions"].get(client_id)
        if normalize(expected) == decision:
            return self._make_outcome(
                milestone_bonus=0.02 if penalty == 0 else 0.0,
                penalty=penalty,
                feedback=f"{client_id} marked as {decision}.",
                outcome_correctness=1.0,
                process_compliance=1.0 if penalty == 0 else 0.5,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        return self._make_outcome(
            penalty=penalty - 0.03,
            feedback=f"{client_id} marked as {decision}, but that choice increases conflict risk.",
            outcome_correctness=0.0,
            process_compliance=1.0,
            deadline_or_latency=1.0,
            safety_or_anti_cheat=1.0,
        )

    def _submit(self, action: Action) -> StepOutcome:
        if action.action_type == "submit_intake":
            ready = len(self._findings["decisions"]) >= len(self._client_lookup)
            if ready:
                return self._make_outcome(
                    milestone_bonus=0.03,
                    feedback="Submission recorded for grading.",
                    outcome_correctness=1.0,
                    process_compliance=1.0,
                    deadline_or_latency=1.0,
                    safety_or_anti_cheat=1.0,
                )
            return self._make_outcome(
                penalty=-0.03,
                feedback="Intake submitted before all client decisions were recorded.",
                outcome_correctness=0.0,
                process_compliance=0.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        if action.action_type == "submit_review":
            ready = (
                len(self._findings["privilege_classifications"]) >= len(self._document_lookup)
                and len(self._findings["recommendations"]) >= len(self._document_lookup)
            )
            if ready:
                return self._make_outcome(
                    milestone_bonus=0.03,
                    feedback="Submission recorded for grading.",
                    outcome_correctness=1.0,
                    process_compliance=1.0,
                    deadline_or_latency=1.0,
                    safety_or_anti_cheat=1.0,
                )
            return self._make_outcome(
                penalty=-0.03,
                feedback="Privilege review submitted before all documents were classified and recommended.",
                outcome_correctness=0.0,
                process_compliance=0.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        required_events = {"EVENT-001", "EVENT-002", "EVENT-003", "EVENT-004", "EVENT-005"}
        action_events = {
            entry.get("event_id")
            for entry in self._findings["actions_taken"]
            if entry.get("event_id") in required_events
        }
        ready = (
            required_events.issubset(action_events)
            and bool(self._findings["ethical_issues_flagged"])
            and bool(self._findings["discovery_response"])
            and bool(self._findings["expert_assessed"])
        )
        if ready:
            return self._make_outcome(
                milestone_bonus=0.03,
                feedback="Submission recorded for grading.",
                outcome_correctness=1.0,
                process_compliance=1.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        return self._make_outcome(
            penalty=-0.03,
            feedback="Triage submitted before the crisis workflow had adequate coverage.",
            outcome_correctness=0.0,
            process_compliance=0.0,
            deadline_or_latency=1.0,
            safety_or_anti_cheat=1.0,
        )

    def _review_document(self, action: Action) -> StepOutcome:
        doc_id = str(action.parameters.get("doc_id", "")).upper()
        document = self._document_lookup.get(doc_id)
        if document is None:
            return self._make_outcome(
                penalty=-0.03,
                feedback=f"Unknown document '{doc_id}'.",
                outcome_correctness=0.0,
                process_compliance=0.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
                flags=["unknown_target"],
            )
        reviewed = self._findings["reviewed_documents"]
        milestone_bonus = 0.02 if doc_id not in reviewed else 0.0
        penalty = -0.01 if doc_id in reviewed else 0.0
        if doc_id not in reviewed:
            reviewed.append(doc_id)
        self._current_content = self._render_document_content(doc_id)
        return self._make_outcome(
            milestone_bonus=milestone_bonus,
            penalty=penalty,
            feedback=f"Reviewed document {doc_id}.",
            outcome_correctness=1.0,
            process_compliance=1.0 if penalty == 0 else 0.5,
            deadline_or_latency=1.0,
            safety_or_anti_cheat=1.0,
        )

    def _classify_privilege(self, action: Action) -> StepOutcome:
        doc_id = str(action.parameters.get("doc_id", "")).upper()
        classification = normalize(action.parameters.get("classification"))
        doctrine = str(action.parameters.get("doctrine", ""))
        valid = {"attorney_client", "work_product", "both", "none", "waived"}
        if doc_id not in PRIVILEGE_GROUND_TRUTH or classification not in valid:
            return self._make_outcome(
                penalty=-0.03,
                feedback="Privilege classification requires a valid doc_id and classification.",
                outcome_correctness=0.0,
                process_compliance=0.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        prerequisite = self._require_reviewed_document(doc_id)
        if prerequisite is not None:
            return prerequisite
        penalty = 0.0
        already_correct = False
        if doc_id in self._findings["privilege_classifications"]:
            old_class = self._findings["privilege_classifications"][doc_id].get("classification")
            if old_class == normalize(PRIVILEGE_GROUND_TRUTH[doc_id]["classification"]):
                already_correct = True
        self._findings["privilege_classifications"][doc_id] = {
            "classification": classification,
            "doctrine": doctrine,
        }
        expected = normalize(PRIVILEGE_GROUND_TRUTH[doc_id]["classification"])
        if classification == expected:
            bonus = 0.0 if already_correct else 0.03
            return self._make_outcome(
                milestone_bonus=bonus,
                penalty=penalty,
                feedback=f"{doc_id} classified correctly as {classification}.",
                outcome_correctness=1.0,
                process_compliance=1.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        if classification in {"attorney_client", "work_product", "both"} and expected in {
            "attorney_client",
            "work_product",
            "both",
        }:
            bonus = 0.0 if already_correct else 0.01
            return self._make_outcome(
                milestone_bonus=bonus,
                penalty=penalty,
                feedback=f"{doc_id} is privileged, but the subtype needs refinement.",
                outcome_correctness=0.5,
                process_compliance=1.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        return self._make_outcome(
            penalty=penalty - 0.03,
            feedback=f"{doc_id} classification is incorrect.",
            outcome_correctness=0.0,
            process_compliance=1.0,
            deadline_or_latency=1.0,
            safety_or_anti_cheat=1.0,
        )

    def _identify_waiver(self, action: Action) -> StepOutcome:
        doc_id = str(action.parameters.get("doc_id", "")).upper()
        waiver_type = normalize(action.parameters.get("waiver_type"))
        if not doc_id or not waiver_type:
            return self._make_outcome(
                penalty=-0.03,
                feedback="Waiver identification requires doc_id and waiver_type.",
                outcome_correctness=0.0,
                process_compliance=0.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        prerequisite = self._require_reviewed_document(doc_id)
        if prerequisite is not None:
            return prerequisite
        entries = self._findings["waivers_identified"]
        expected = normalize(WAIVER_EVENTS.get(doc_id, ""))
        already_identified = any(
            entry.get("doc_id") == doc_id and entry.get("waiver_type") == expected
            for entry in entries
        )
        entries[:] = [entry for entry in entries if entry.get("doc_id") != doc_id]
        entries.append(
            {
                "doc_id": doc_id,
                "waiver_type": waiver_type,
                "explanation": str(action.parameters.get("explanation", "")),
            }
        )
        if waiver_type == expected:
            bonus = 0.0 if already_identified else 0.03
            return self._make_outcome(
                milestone_bonus=bonus,
                feedback=f"Waiver risk correctly identified for {doc_id}.",
                outcome_correctness=1.0,
                process_compliance=1.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        return self._make_outcome(
            penalty=-0.03,
            feedback=f"Waiver call for {doc_id} does not match the ground truth.",
            outcome_correctness=0.0,
            process_compliance=1.0,
            deadline_or_latency=1.0,
            safety_or_anti_cheat=1.0,
        )

    def _identify_exception(self, action: Action) -> StepOutcome:
        doc_id = str(action.parameters.get("doc_id", "")).upper()
        exception_type = normalize(action.parameters.get("exception_type"))
        if not doc_id or not exception_type:
            return self._make_outcome(
                penalty=-0.03,
                feedback="Exception identification requires doc_id and exception_type.",
                outcome_correctness=0.0,
                process_compliance=0.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        prerequisite = self._require_reviewed_document(doc_id)
        if prerequisite is not None:
            return prerequisite
        entries = self._findings["exceptions_identified"]
        entries[:] = [entry for entry in entries if entry.get("doc_id") != doc_id]
        entries.append(
            {
                "doc_id": doc_id,
                "exception_type": exception_type,
                "explanation": str(action.parameters.get("explanation", "")),
            }
        )
        expected = normalize(PRIVILEGE_GROUND_TRUTH.get(doc_id, {}).get("exception"))
        if expected != "none" and exception_type == expected:
            return self._make_outcome(
                milestone_bonus=0.02,
                feedback=f"Privilege exception correctly identified for {doc_id}.",
                outcome_correctness=1.0,
                process_compliance=1.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        return self._make_outcome(
            penalty=-0.02,
            feedback=f"Exception assessment for {doc_id} is not supported by the scenario.",
            outcome_correctness=0.0,
            process_compliance=1.0,
            deadline_or_latency=1.0,
            safety_or_anti_cheat=1.0,
        )

    def _recommend_action(self, action: Action) -> StepOutcome:
        doc_id = str(action.parameters.get("doc_id", "")).upper()
        recommendation = normalize(action.parameters.get("action"))
        if doc_id not in PRIVILEGE_GROUND_TRUTH or not recommendation:
            return self._make_outcome(
                penalty=-0.03,
                feedback="Recommendation requires doc_id and action.",
                outcome_correctness=0.0,
                process_compliance=0.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        prerequisite = self._require_reviewed_document(doc_id)
        if prerequisite is not None:
            return prerequisite
        self._findings["recommendations"][doc_id] = {
            "action": recommendation,
            "reasoning": str(action.parameters.get("reasoning", "")),
        }
        expected = normalize(PRIVILEGE_GROUND_TRUTH[doc_id]["action"])
        if recommendation == expected:
            return self._make_outcome(
                milestone_bonus=0.02,
                feedback=f"Production recommendation for {doc_id} is correct.",
                outcome_correctness=1.0,
                process_compliance=1.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        return self._make_outcome(
            penalty=-0.02,
            feedback=f"Production recommendation for {doc_id} is misaligned with the privilege posture.",
            outcome_correctness=0.0,
            process_compliance=1.0,
            deadline_or_latency=1.0,
            safety_or_anti_cheat=1.0,
        )

    def _review_event(self, action: Action) -> StepOutcome:
        event_id = str(action.parameters.get("event_id", "")).upper()
        event = self._event_lookup.get(event_id)
        if event is None:
            return self._make_outcome(
                penalty=-0.03,
                feedback=f"Unknown event '{event_id}'.",
                outcome_correctness=0.0,
                process_compliance=0.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
                flags=["unknown_target"],
            )
        reviewed = self._findings["reviewed_events"]
        bonus = 0.02 if event_id not in reviewed else 0.0
        penalty = -0.01 if event_id in reviewed else 0.0
        if event_id not in reviewed:
            reviewed.append(event_id)
        self._findings["actions_taken"].append(
            {"event_id": event_id, "action": "review_event", "step": self._step_count}
        )
        self._current_content = self._render_event_content(event_id)
        return self._make_outcome(
            milestone_bonus=bonus,
            penalty=penalty,
            feedback=f"Reviewed crisis event {event_id}.",
            outcome_correctness=1.0,
            process_compliance=1.0 if penalty == 0 else 0.5,
            deadline_or_latency=1.0,
            safety_or_anti_cheat=1.0,
        )

    def _issue_litigation_hold(self, action: Action) -> StepOutcome:
        prerequisite = self._require_reviewed_event("EVENT-001")
        if prerequisite is not None:
            return prerequisite
        scope = str(action.parameters.get("scope", ""))
        custodians = action.parameters.get("custodians", [])
        if isinstance(custodians, str):
            custodians = [item.strip() for item in custodians.split(",") if item.strip()]
        if not scope or not custodians:
            return self._make_outcome(
                penalty=-0.03,
                feedback="Litigation hold requires scope and custodians.",
                outcome_correctness=0.0,
                process_compliance=0.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        self._findings["deadlines_met"]["EVENT-001"] = {
            "step": self._step_count,
            "scope": scope,
            "custodians": custodians,
        }
        self._findings["actions_taken"].append(
            {"event_id": "EVENT-001", "action": "issue_litigation_hold", "step": self._step_count}
        )
        deadline = self._event_deadline("EVENT-001")
        coverage = first_matching(custodians, ["morton", "ames", "wong", "liu", "park"])
        coverage_bonus = 0.02 if coverage else 0.0
        if self._step_count <= deadline:
            return self._make_outcome(
                milestone_bonus=0.05 + coverage_bonus,
                feedback="Litigation hold issued before the preservation deadline.",
                outcome_correctness=1.0,
                process_compliance=1.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        return self._make_outcome(
            penalty=-0.08,
            feedback="Litigation hold was issued after the preservation deadline.",
            outcome_correctness=0.5,
            process_compliance=1.0,
            deadline_or_latency=0.0,
            safety_or_anti_cheat=1.0,
            flags=["late_deadline_action"],
        )

    def _file_motion(self, action: Action) -> StepOutcome:
        motion_type = normalize(action.parameters.get("motion_type"))
        court = str(action.parameters.get("court", ""))
        if not motion_type:
            return self._make_outcome(
                penalty=-0.03,
                feedback="Motion filing requires motion_type.",
                outcome_correctness=0.0,
                process_compliance=0.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        target_event = self._target_event_for_motion(motion_type)
        if target_event:
            prerequisite = self._require_reviewed_event(target_event)
            if prerequisite is not None:
                return prerequisite
        if not target_event:
            self._findings["actions_taken"].append(
                {"event_id": "UNMAPPED", "action": "file_motion", "step": self._step_count}
            )
            return self._make_outcome(
                penalty=-0.02,
                feedback="The motion was filed, but it does not resolve a scored crisis event.",
                outcome_correctness=0.0,
                process_compliance=1.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        self._findings["deadlines_met"][target_event] = {
            "step": self._step_count,
            "motion_type": motion_type,
            "court": court,
        }
        self._findings["actions_taken"].append(
            {"event_id": target_event, "action": "file_motion", "step": self._step_count}
        )
        deadline = self._event_deadline(target_event)
        if self._step_count <= deadline:
            return self._make_outcome(
                milestone_bonus=0.05,
                feedback=f"Motion work for {target_event} was filed on time.",
                outcome_correctness=1.0,
                process_compliance=1.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        return self._make_outcome(
            penalty=-0.06,
            feedback=f"Motion work for {target_event} was filed late.",
            outcome_correctness=0.5,
            process_compliance=1.0,
            deadline_or_latency=0.0,
            safety_or_anti_cheat=1.0,
            flags=["late_deadline_action"],
        )

    def _respond_discovery(self, action: Action) -> StepOutcome:
        response_type = normalize(action.parameters.get("response_type"))
        objections = str(action.parameters.get("objections", ""))
        if not response_type:
            return self._make_outcome(
                penalty=-0.03,
                feedback="Discovery response requires response_type.",
                outcome_correctness=0.0,
                process_compliance=0.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        # ── Privilege-waiver trap fires BEFORE the review prerequisite ──
        # Careless production is catastrophic whether the agent has seen
        # EVENT-003 or not — that is the whole point of the adversarial trap.
        if response_type == "produce" and not objections:
            self._findings["discovery_response"] = {
                "step": self._step_count,
                "response_type": response_type,
                "objections": objections,
            }
            self._findings["actions_taken"].append(
                {"event_id": "EVENT-003", "action": "respond_discovery", "step": self._step_count}
            )
            self._ethical_alerts.append(
                "Producing advocate communications without objections risks broad privilege waiver."
            )
            deadline = self._event_deadline("EVENT-003")
            return self._make_outcome(
                penalty=-0.12,
                feedback="Discovery response risks waiving privilege by producing without objections.",
                outcome_correctness=0.0,
                process_compliance=1.0,
                deadline_or_latency=0.0 if self._step_count > deadline else 1.0,
                safety_or_anti_cheat=0.0,
                flags=["privilege_waiver_trap"],
            )
        # ── Non-catastrophic responses require a prior review ──
        prerequisite = self._require_reviewed_event("EVENT-003")
        if prerequisite is not None:
            return prerequisite
        self._findings["discovery_response"] = {
            "step": self._step_count,
            "response_type": response_type,
            "objections": objections,
        }
        self._findings["deadlines_met"]["EVENT-003"] = {
            "step": self._step_count,
            "response_type": response_type,
        }
        self._findings["actions_taken"].append(
            {"event_id": "EVENT-003", "action": "respond_discovery", "step": self._step_count}
        )
        deadline = self._event_deadline("EVENT-003")
        bonus = 0.03 if self._step_count <= deadline else 0.0
        penalty = -0.05 if self._step_count > deadline else 0.0
        if response_type in {"privilege_log", "object", "partial_produce"} and any(
            term in normalize(objections) for term in ("privilege", "section 126", "section 129", "advocate")
        ):
            return self._make_outcome(
                milestone_bonus=bonus + 0.03,
                penalty=penalty,
                feedback="Discovery response preserved privilege and documented objections.",
                outcome_correctness=1.0,
                process_compliance=1.0,
                deadline_or_latency=1.0 if self._step_count <= deadline else 0.0,
                safety_or_anti_cheat=1.0,
            )
        return self._make_outcome(
            milestone_bonus=bonus,
            penalty=penalty - 0.02,
            feedback="Discovery response was recorded, but the privilege rationale is weak.",
            outcome_correctness=0.5,
            process_compliance=1.0,
            deadline_or_latency=1.0 if self._step_count <= deadline else 0.0,
            safety_or_anti_cheat=0.5,
        )

    def _assess_expert(self, action: Action) -> StepOutcome:
        prerequisite = self._require_reviewed_event("EVENT-005")
        if prerequisite is not None:
            return prerequisite
        qualification = str(action.parameters.get("qualification", ""))
        if not qualification:
            return self._make_outcome(
                penalty=-0.03,
                feedback="Expert assessment requires qualification details.",
                outcome_correctness=0.0,
                process_compliance=0.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        self._findings["expert_assessed"] = {
            "expert_id": str(action.parameters.get("expert_id", "EXPERT")),
            "qualification": qualification,
            "step": self._step_count,
        }
        self._findings["actions_taken"].append(
            {"event_id": "EVENT-005", "action": "assess_expert", "step": self._step_count}
        )
        if any(term in normalize(qualification) for term in ("special skill", "science", "toxicology", "section 45")):
            return self._make_outcome(
                milestone_bonus=0.03,
                feedback="Expert qualification analysis addresses the key admissibility factors.",
                outcome_correctness=1.0,
                process_compliance=1.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        return self._make_outcome(
            penalty=-0.01,
            feedback="Expert qualification analysis is too shallow for Section 45 review.",
            outcome_correctness=0.5,
            process_compliance=1.0,
            deadline_or_latency=1.0,
            safety_or_anti_cheat=1.0,
        )

    def _flag_adversarial(self, action: Action) -> StepOutcome:
        item_id = str(action.parameters.get("item_id", "")).upper()
        threat_type = str(action.parameters.get("threat_type", ""))
        if not item_id or not threat_type:
            return self._make_outcome(
                penalty=-0.03,
                feedback="Adversarial flagging requires item_id and threat_type.",
                outcome_correctness=0.0,
                process_compliance=0.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        prerequisite = self._require_reviewed_event(item_id)
        if prerequisite is not None:
            return prerequisite
        flags = self._findings["adversarial_flagged"]
        already_flagged = any(entry.get("item_id") == item_id for entry in flags)
        flags[:] = [entry for entry in flags if entry.get("item_id") != item_id]
        flags.append(
            {
                "item_id": item_id,
                "threat_type": threat_type,
                "explanation": str(action.parameters.get("explanation", "")),
            }
        )
        if item_id in CRISIS_GROUND_TRUTH["adversarial_items"]:
            bonus = 0.0 if already_flagged else 0.03
            return self._make_outcome(
                milestone_bonus=bonus,
                feedback=f"Adversarial pattern correctly flagged for {item_id}.",
                outcome_correctness=1.0,
                process_compliance=1.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        return self._make_outcome(
            penalty=-0.02,
            feedback=f"{item_id} is not a scored adversarial event.",
            outcome_correctness=0.0,
            process_compliance=1.0,
            deadline_or_latency=1.0,
            safety_or_anti_cheat=1.0,
        )

    def _flag_ethical_issue(self, action: Action) -> StepOutcome:
        prerequisite = self._require_reviewed_event("EVENT-004")
        if prerequisite is not None:
            return prerequisite
        issue_type = str(action.parameters.get("issue_type", ""))
        resolution = str(action.parameters.get("resolution", ""))
        if not issue_type:
            return self._make_outcome(
                penalty=-0.03,
                feedback="Ethical issue flagging requires issue_type.",
                outcome_correctness=0.0,
                process_compliance=0.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        entries = self._findings["ethical_issues_flagged"]
        entries[:] = [entry for entry in entries if entry.get("event_id") != "EVENT-004"]
        entries.append(
            {
                "event_id": "EVENT-004",
                "issue_type": issue_type,
                "affected_clients": str(action.parameters.get("affected_clients", "")),
                "resolution": resolution,
            }
        )
        self._findings["actions_taken"].append(
            {"event_id": "EVENT-004", "action": "flag_ethical_issue", "step": self._step_count}
        )
        keywords = ("withdraw", "screen", "consent", "disclose", "rule 33", "former client")
        if any(keyword in normalize(resolution) for keyword in keywords):
            return self._make_outcome(
                milestone_bonus=0.05,
                feedback="Ethical conflict surfaced with a defensible mitigation plan.",
                outcome_correctness=1.0,
                process_compliance=1.0,
                deadline_or_latency=1.0,
                safety_or_anti_cheat=1.0,
            )
        return self._make_outcome(
            milestone_bonus=0.02,
            feedback="Ethical conflict was flagged, but the resolution needs stronger mitigation language.",
            outcome_correctness=0.5,
            process_compliance=1.0,
            deadline_or_latency=1.0,
            safety_or_anti_cheat=1.0,
        )

    def _noop(self, action: Action) -> StepOutcome:
        del action
        return self._make_outcome(
            feedback="No action taken.",
            outcome_correctness=0.0,
            process_compliance=0.25,
            deadline_or_latency=0.25,
            safety_or_anti_cheat=0.5,
        )


class LexCrisisSessionManager:
    """Track episode-specific engines so concurrent sessions do not collide."""

    def __init__(self) -> None:
        self._engines: Dict[str, LexCrisisEngine] = {}
        self._latest_episode_id: Optional[str] = None

    def reset_episode(
        self,
        *,
        task_id: Optional[str],
        seed: Optional[int],
        episode_id: Optional[str],
        episode_config: Optional[Dict[str, Any]] = None,
    ) -> Observation:
        target_episode_id = episode_id or str(uuid4())
        engine = self._engines.get(target_episode_id)
        if engine is None:
            engine = LexCrisisEngine()
            self._engines[target_episode_id] = engine
        observation = engine.reset(
            task_id=task_id,
            seed=seed,
            episode_id=target_episode_id,
            episode_config=episode_config,
        )
        self._latest_episode_id = target_episode_id
        return observation

    def get_engine(self, episode_id: Optional[str] = None) -> Optional[LexCrisisEngine]:
        target = episode_id or self._latest_episode_id
        if target is None:
            return None
        return self._engines.get(target)


_SESSIONS = LexCrisisSessionManager()


class LexCrisisEnvironment(Environment[Action, Observation, EnvironmentState]):
    """OpenEnv-compatible wrapper around LexCrisis engines."""

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self) -> None:
        super().__init__()
        self._episode_id: Optional[str] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        episode_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Observation:
        del kwargs
        observation = _SESSIONS.reset_episode(
            task_id=task_id,
            seed=seed,
            episode_id=episode_id,
            episode_config=episode_config,
        )
        self._episode_id = observation.metadata.get("episode_id")
        return observation

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        del timeout_s, kwargs
        engine = _SESSIONS.get_engine(self._episode_id) or _SESSIONS.get_engine()
        if engine is None:
            observation = self.reset(task_id="task_1")
            self._episode_id = observation.metadata.get("episode_id")
            engine = _SESSIONS.get_engine(self._episode_id)
        observation, _, _, _ = engine.step(action)
        self._episode_id = engine.episode_id
        return observation

    @property
    def state(self) -> EnvironmentState:
        engine = _SESSIONS.get_engine(self._episode_id) or _SESSIONS.get_engine()
        if engine is None:
            fallback = self.reset(task_id="task_1")
            engine = _SESSIONS.get_engine(fallback.metadata.get("episode_id"))
        return engine.state()

    @property
    def last_score(self) -> float:
        engine = _SESSIONS.get_engine(self._episode_id) or _SESSIONS.get_engine()
        return engine.last_score if engine is not None else _SCORE_FLOOR

    @property
    def episode_id(self) -> str:
        engine = _SESSIONS.get_engine(self._episode_id) or _SESSIONS.get_engine()
        return engine.episode_id if engine is not None else ""

    def episode_info(self, episode_id: Optional[str] = None) -> Dict[str, Any]:
        engine = _SESSIONS.get_engine(episode_id or self._episode_id) or _SESSIONS.get_engine()
        return engine.episode_info() if engine is not None else {}

    def get_metadata(self) -> EnvironmentMetadata:
        readme_path = Path(__file__).resolve().parents[1] / "README.md"
        readme_content = None
        if readme_path.exists():
            readme_content = readme_path.read_text(encoding="utf-8")
        return EnvironmentMetadata(
            name="LexCrisis",
            description=(
                "Law-focused benchmark for legal operations incident response in "
                "high-stakes product-liability litigation."
            ),
            readme_content=readme_content,
            version=BENCHMARK_VERSION,
            author="OpenEnv Hackathon Submission",
        )

    def close(self) -> None:
        """Environment instances are light wrappers over the shared session manager."""
        return None
