#!/usr/bin/env python3
"""Self-improvement strategy for the LexCrisis OpenEnv agent.

This script demonstrates three post-training strategies to push LexCrisis agents
beyond the oracle reference scores:

  Phase 1 — SFT Data Generation
    Rolls out oracle reference trajectories and records (observation, action) pairs as
    fine-tuning examples in ShareGPT format. Feed these to any SFT framework
    (TRL, LLaMA-Factory, OpenRLHF) to teach a base model the action schema
    and correct legal reasoning before RL begins.

  Phase 2 — Adversarial Curriculum via Task Augmentation
    Generates additional training episodes by randomising deadline_step values
    and shuffling document/event order. Prevents overfitting to the fixed
    reference trajectory and forces the agent to generalise its triage policy.

  Phase 3 — LLM-as-Judge Failure Analysis
    For episodes that score below a threshold, produces structured feedback
    explaining exactly which sub-component caused the score drop. This feedback
    is appended to the next rollout's prompt, creating a hindsight learning loop.

Usage:
    python self_improve.py                    # run all phases, default config
    python self_improve.py --phase sft        # SFT data generation only
    python self_improve.py --phase curriculum # augmented episodes only
    python self_improve.py --phase judge      # failure analysis only
    python self_improve.py --tasks task_3     # run on specific tasks

Output files written to ./outputs/self_improve/
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lexcrisis_env.env import LexCrisisEngine
from lexcrisis_env.graders import GRADERS, GROUND_TRUTH
from lexcrisis_env.models import Action
from lexcrisis_env.prompting import SYSTEM_PROMPT
from lexcrisis_env.tasks import (
    CRISIS_EVENTS,
    PRIVILEGE_DOCUMENTS,
    SCRIPTED_BASELINES,
    TASK_DEFINITIONS,
)


OUTPUT_DIR = Path("outputs") / "self_improve"
SCORE_THRESHOLD = 0.75  # Episodes below this threshold trigger failure analysis


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    sys.stderr.write(f"# {msg}\n")
    sys.stderr.flush()


def _run_episode(
    task_id: str,
    actions: List[Dict[str, Any]],
    seed: Optional[int] = None,
    episode_config: Optional[Dict[str, Any]] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Execute a sequence of actions and return (final_score, step_log)."""
    engine = LexCrisisEngine()
    engine.reset(task_id=task_id, seed=seed, episode_config=episode_config)
    step_log: List[Dict[str, Any]] = []

    for raw in actions:
        action = Action.model_validate(raw)
        obs, reward, done, info = engine.step(action)
        step_log.append(
            {
                "action": raw,
                "reward": round(reward, 4),
                "score": info.get("score", 0.0),
                "feedback": obs.feedback,
                "done": done,
                "verifier_signals": info.get("reward_breakdown", {}).get("verifier_signals", {}),
            }
        )
        if done:
            break

    return engine.last_score, step_log


# ──────────────────────────────────────────────────────────────────────
# Phase 1 — SFT Data Generation
# ──────────────────────────────────────────────────────────────────────

def generate_sft_data(task_ids: List[str]) -> List[Dict[str, Any]]:
    """Roll out oracle reference trajectories and emit ShareGPT-format fine-tuning examples.

    Each step produces one (system, user, assistant) example where:
      - system  = the shared production system prompt
      - user    = the full observation JSON at that step
      - assistant = the correct action JSON the agent should output

    These examples teach a base model:
      1. The JSON action schema
      2. The correct legal reasoning steps
      3. The relationship between observation content and action selection
    """
    examples: List[Dict[str, Any]] = []

    for task_id in task_ids:
        _log(f"Generating SFT data for {task_id} from oracle reference actions...")
        engine = LexCrisisEngine()
        obs = engine.reset(task_id=task_id)

        for step_idx, raw_action in enumerate(SCRIPTED_BASELINES[task_id], 1):
            obs_dict = obs.model_dump(mode="json")

            # Build the user turn exactly as inference.py does
            user_content = json.dumps(
                {
                    "task_id": task_id,
                    "step": step_idx,
                    "steps_remaining": obs_dict.get("max_steps", 0) - step_idx,
                    "feedback_from_last_action": obs_dict.get("feedback", ""),
                    "available_actions": obs_dict.get("available_actions", []),
                    "findings_so_far": {
                        k: v for k, v in obs_dict.get("findings", {}).items() if v
                    },
                    "active_deadlines": obs_dict.get("active_deadlines", []),
                    "revealed_content": obs_dict.get("current_content"),
                    "selectable_items": obs_dict.get("documents", []),
                },
                indent=2,
            )

            examples.append(
                {
                    "conversations": [
                        {"from": "system", "value": SYSTEM_PROMPT},
                        {"from": "human", "value": user_content},
                        {"from": "gpt", "value": json.dumps(raw_action)},
                    ],
                    "metadata": {
                        "task_id": task_id,
                        "step": step_idx,
                        "action_type": raw_action["action_type"],
                    },
                }
            )

            # Advance the engine
            action = Action.model_validate(raw_action)
            obs, _, done, _ = engine.step(action)
            if done:
                break

    _log(f"Generated {len(examples)} SFT examples across {len(task_ids)} tasks.")
    return examples


# ──────────────────────────────────────────────────────────────────────
# Phase 2 — Adversarial Curriculum (Task Augmentation)
# ──────────────────────────────────────────────────────────────────────

def generate_curriculum_episodes(
    task_ids: List[str],
    n_variants: int = 5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate augmented training episodes by perturbing the environment data.

    Augmentation strategies applied:
      1. Deadline randomisation (±3 steps) — forces generalised triage policy
      2. Document/event order shuffling — prevents position-based shortcuts
      3. Custodian name variation — tests generalisation of hold-issuance policy

    Each variant is run with the oracle reference policy.
    The resulting (obs, action, reward) tuples are additional RL training data.
    """
    episodes: List[Dict[str, Any]] = []

    for task_id in task_ids:
        if task_id != "task_3":
            # Curriculum augmentation has highest impact on the deadline-sensitive task
            continue

        for variant_idx in range(n_variants):
            _log(f"Generating curriculum variant {variant_idx + 1}/{n_variants} for {task_id}...")

            # Perturb deadline steps within ±3 of ground truth
            variant_seed = seed + variant_idx
            preview_engine = LexCrisisEngine()
            preview_engine.reset(task_id=task_id, seed=variant_seed)
            score, step_log = _run_episode(
                task_id,
                SCRIPTED_BASELINES[task_id],
                seed=variant_seed,
            )

            episodes.append(
                {
                    "variant_id": f"{task_id}_curriculum_{variant_idx}",
                    "task_id": task_id,
                    "seed": variant_seed,
                    "episode_config": preview_engine.episode_config,
                    "perturbed_deadlines": preview_engine.episode_config.get("deadline_overrides", {}),
                    "final_score": score,
                    "steps": step_log,
                    "augmentation": "seeded_randomisation",
                }
            )

    _log(f"Generated {len(episodes)} curriculum episodes.")
    return episodes


# ──────────────────────────────────────────────────────────────────────
# Phase 3 — LLM-as-Judge Failure Analysis
# ──────────────────────────────────────────────────────────────────────

def analyse_failures(
    task_ids: List[str],
    threshold: float = SCORE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Run oracle reference trajectories and produce structured failure analysis for low-scoring episodes.

    For each task that scores below `threshold`, the analyser:
      1. Computes which grader sub-component caused the biggest score gap
      2. Identifies the step(s) where points were lost
      3. Generates a structured feedback message that can be injected into
         the next training episode's system prompt or observation

    This implements a lightweight "hindsight feedback loop" without requiring
    an external LLM — the deterministic graders provide ground truth for the analysis.
    """
    analyses: List[Dict[str, Any]] = []

    for task_id in task_ids:
        _log(f"Running failure analysis for {task_id}...")
        score, step_log = _run_episode(task_id, SCRIPTED_BASELINES[task_id])

        if score >= threshold:
            _log(f"  {task_id} score {score:.4f} is above threshold {threshold}. Skipping.")
            continue

        _log(f"  {task_id} score {score:.4f} is below threshold. Analysing...")

        # Find the steps with the largest negative rewards
        negative_steps = sorted(
            [s for s in step_log if s["reward"] < 0],
            key=lambda s: s["reward"],
        )

        # Build structured feedback for each failure point
        failure_messages: List[str] = []
        for s in negative_steps[:3]:  # top 3 worst steps
            action_type = s["action"].get("action_type", "unknown")
            failure_messages.append(
                f"Action '{action_type}' produced reward {s['reward']:.4f}. "
                f"Environment feedback: {s['feedback']}"
            )

        # Compute sub-component scores to identify the weakest area
        engine = LexCrisisEngine()
        engine.reset(task_id=task_id)
        for raw in SCRIPTED_BASELINES[task_id]:
            action = Action.model_validate(raw)
            _, _, done, _ = engine.step(action)
            if done:
                break

        grader = GRADERS[task_id]
        truth = GROUND_TRUTH[task_id]
        final_findings = copy.deepcopy(engine._findings)  # type: ignore[attr-defined]

        # Task-specific sub-component diagnosis
        diagnosis = _diagnose(task_id, final_findings, truth)

        analyses.append(
            {
                "task_id": task_id,
                "final_score": score,
                "threshold": threshold,
                "failure_steps": failure_messages,
                "diagnosis": diagnosis,
                "hindsight_prompt": _build_hindsight_prompt(task_id, failure_messages, diagnosis),
            }
        )

    _log(f"Produced {len(analyses)} failure analyses.")
    return analyses


def _diagnose(
    task_id: str,
    findings: Dict[str, Any],
    truth: Dict[str, Any],
) -> str:
    """Produce a plain-English diagnosis of the weakest grader sub-component."""
    if task_id == "task_1":
        decisions = findings.get("decisions", {})
        correct_decisions = truth.get("correct_decisions", {})
        wrong = [
            cid
            for cid, expected in correct_decisions.items()
            if decisions.get(cid) != expected
        ]
        if wrong:
            return f"Incorrect intake decisions for: {', '.join(wrong)}. Review their adverse relationships."
        citations = findings.get("rule_citations", [])
        if len(citations) < len(truth.get("conflict_rules", {})):
            return "Missing rule citations. Every identified conflict pair needs a cite_rule action."
        return "Conflict pair detection may be incomplete. Check all client relationship combinations."

    if task_id == "task_2":
        classifications = findings.get("privilege_classifications", {})
        wrong_docs = [
            doc_id
            for doc_id, t in truth.items()
            if classifications.get(doc_id, {}).get("classification") != t["classification"]
        ]
        if wrong_docs:
            return f"Incorrect privilege classification for: {', '.join(wrong_docs)}."
        waivers = {e["doc_id"] for e in findings.get("waivers_identified", [])}
        missing_waivers = {"DOC-006", "DOC-007"} - waivers
        if missing_waivers:
            return f"Missing waiver identification for: {', '.join(missing_waivers)}."
        return "Recommendation actions may be missing or incorrect for some documents."

    # task_3
    deadlines = findings.get("deadlines_met", {})
    required = {"EVENT-001": 6, "EVENT-002": 9, "EVENT-003": 12}
    missed = [e for e in required if e not in deadlines]
    if missed:
        return (
            f"Missed deadlines: {', '.join(missed)}. "
            f"These must be addressed before their deadline_step."
        )
    if not findings.get("adversarial_flagged"):
        return (
            "EVENT-003 adversarial trap was not flagged. "
            "Call flag_adversarial with threat_type 'privilege_trap' before respond_discovery."
        )
    discovery = findings.get("discovery_response", {})
    if discovery.get("response_type") == "produce" and not discovery.get("objections"):
        return (
            "CRITICAL: respond_discovery used 'produce' without objections. "
            "This triggers a -0.12 privilege waiver penalty. "
            "Always use 'privilege_log' with BSA Sections 132/134 objections."
        )
    return "Action ordering sub-score may be low. Follow priority: EVENT-001 → 004 → 002 → 003 → 005."


def _build_hindsight_prompt(
    task_id: str,
    failure_steps: List[str],
    diagnosis: str,
) -> str:
    """Build a hindsight feedback string to inject into the next training prompt."""
    lines = [
        f"[HINDSIGHT FEEDBACK for {task_id}]",
        f"Diagnosis: {diagnosis}",
    ]
    if failure_steps:
        lines.append("Failed steps:")
        for msg in failure_steps:
            lines.append(f"  - {msg}")
    lines.append(
        "In the next episode, prioritise fixing the identified weakness before "
        "pursuing other actions."
    )
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LexCrisis self-improvement pipeline")
    parser.add_argument(
        "--phase",
        choices=["sft", "curriculum", "judge", "all"],
        default="all",
        help="Which phase to run (default: all)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(TASK_DEFINITIONS.keys()),
        default=list(TASK_DEFINITIONS.keys()),
        help="Which tasks to include (default: all)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=SCORE_THRESHOLD,
        help=f"Failure analysis threshold (default: {SCORE_THRESHOLD})",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=5,
        help="Number of curriculum variants per task (default: 5)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log(f"Output directory: {OUTPUT_DIR}")
    _log(f"Tasks: {args.tasks}")
    _log(f"Phase: {args.phase}")

    if args.phase in ("sft", "all"):
        _log("\n=== Phase 1: SFT Data Generation ===")
        sft_data = generate_sft_data(args.tasks)
        out = OUTPUT_DIR / "sft_examples.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for ex in sft_data:
                f.write(json.dumps(ex) + "\n")
        _log(f"Wrote {len(sft_data)} SFT examples to {out}")

    if args.phase in ("curriculum", "all"):
        _log("\n=== Phase 2: Adversarial Curriculum ===")
        episodes = generate_curriculum_episodes(args.tasks, n_variants=args.variants)
        out = OUTPUT_DIR / "curriculum_episodes.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for ep in episodes:
                f.write(json.dumps(ep) + "\n")
        _log(f"Wrote {len(episodes)} curriculum episodes to {out}")

    if args.phase in ("judge", "all"):
        _log("\n=== Phase 3: Failure Analysis ===")
        analyses = analyse_failures(args.tasks, threshold=args.threshold)
        out = OUTPUT_DIR / "failure_analyses.json"
        out.write_text(json.dumps(analyses, indent=2), encoding="utf-8")
        _log(f"Wrote {len(analyses)} analyses to {out}")
        for a in analyses:
            _log(f"\n  {a['task_id']} (score={a['final_score']:.4f}):")
            _log(f"    {a['diagnosis']}")
            _log(f"    Hindsight: {a['hindsight_prompt'][:200]}...")

    _log("\nSelf-improvement pipeline complete.")


if __name__ == "__main__":
    main()
