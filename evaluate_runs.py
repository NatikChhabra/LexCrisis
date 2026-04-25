#!/usr/bin/env python3
"""Evaluate LexCrisis policy traces and write reproducible JSON artifacts."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

from lexcrisis_env.env import LexCrisisEngine
from lexcrisis_env.graders import GRADER_BREAKDOWNS, GROUND_TRUTH
from lexcrisis_env.models import Action
from lexcrisis_env.tasks import CRISIS_GROUND_TRUTH, SCRIPTED_BASELINES, TASK_DEFINITIONS

OUTPUT_DIR = Path("outputs") / "evals"

def _debug_enabled() -> bool:
    return os.getenv("LEXCRISIS_DEBUG") == "1"


def _debug_log(message: str, data: Dict[str, Any]) -> None:
    if _debug_enabled():
        print(f"[debug] {message}: {json.dumps(data, sort_keys=True)}")


def resolve_trace_file(trace_dir: Path, task_id: str, suite: str, seed: Optional[int]) -> Path:
    """Resolve the concrete trace file path from candidate naming schemes."""
    seed_label = "fixed" if seed is None else str(seed)
    candidates = [
        trace_dir / f"{task_id}__{suite}__{seed_label}.json",
        trace_dir / f"{task_id}__{seed_label}.json",
        trace_dir / f"{task_id}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No trace file found for task={task_id}, suite={suite}, seed={seed_label} in {trace_dir}"
    )


def load_trace_actions(trace_file: Path, task_id: str, suite: str, seed: Optional[int]) -> List[Dict[str, Any]]:
    """Load a resolved trace file from disk."""
    actions = json.loads(trace_file.read_text(encoding="utf-8"))
    _debug_log(
        "Loaded trace file",
        {
            "task_id": task_id,
            "suite": suite,
            "seed": "fixed" if seed is None else str(seed),
            "path": str(trace_file),
            "action_count": len(actions),
        },
    )
    return actions


def run_episode(
    run_name: str,
    run_kind: str,
    run_value: str,
    task_id: str,
    suite: str,
    seed: Optional[int],
    pass_threshold: float,
) -> Dict[str, Any]:
    """Run one episode and collect rich metrics."""

    trace_source = ""
    if run_kind == "scripted":
        actions = SCRIPTED_BASELINES[task_id]
    elif run_kind == "trace_dir":
        trace_path = resolve_trace_file(Path(run_value), task_id, suite, seed)
        trace_source = str(trace_path)
        actions = load_trace_actions(trace_path, task_id, suite, seed)
    else:
        raise ValueError(f"Unsupported run kind: {run_kind}")

    engine = LexCrisisEngine()
    observation = engine.reset(task_id=task_id, seed=seed)
    _debug_log(
        "Episode start configuration",
        {
            "run_name": run_name,
            "run_kind": run_kind,
            "task_id": task_id,
            "suite": suite,
            "seed": seed,
            "action_count": len(actions),
        },
    )
    if run_kind == "trace_dir":
        is_exact_scripted = actions == SCRIPTED_BASELINES[task_id]
        _debug_log(
            "Trace-vs-scripted equality check",
            {
                "run_name": run_name,
                "task_id": task_id,
                "suite": suite,
                "seed": seed,
                "exact_scripted_match": is_exact_scripted,
            },
        )
    rewards: List[float] = []
    successful_steps = 0
    invalid_actions = 0
    review_failures = 0
    privilege_trap_hits = 0
    unique_trace: List[str] = []

    for raw in actions:
        action = Action.model_validate(raw)
        observation, reward, done, info = engine.step(action)
        rewards.append(reward)
        if reward > 0:
            successful_steps += 1
        breakdown = info.get("reward_breakdown", {})
        flags = breakdown.get("flags", [])
        if any(flag in {"unknown_action", "unknown_target"} for flag in flags):
            invalid_actions += 1
        if "review_prerequisite_missing" in flags:
            review_failures += 1
        if "privilege_waiver_trap" in flags:
            privilege_trap_hits += 1
        unique_trace.append(f"{action.action_type}|{json.dumps(action.parameters, sort_keys=True)}")
        if done:
            break

    findings = getattr(engine, "_findings")
    truth = copy.deepcopy(GROUND_TRUTH[task_id])
    if task_id == "task_3":
        truth["deadlines"] = engine.episode_config.get("deadline_overrides", truth["deadlines"])
        truth["deadlines"] = {
            event_id: {
                "deadline_step": deadline,
                "required_action": CRISIS_GROUND_TRUTH["deadlines"][event_id]["required_action"],
            }
            for event_id, deadline in truth["deadlines"].items()
        }
    breakdown = GRADER_BREAKDOWNS[task_id](findings, truth)
    total_steps = max(1, len(rewards))
    final_score = engine.last_score
    done = engine.state().done
    last_verifier = engine.episode_info().get("verifier_columns", {})
    _debug_log(
        "Verifier columns available at runtime",
        {
            "run_name": run_name,
            "task_id": task_id,
            "suite": suite,
            "seed": seed,
            "final_score": round(final_score, 4),
            "verifier_columns": last_verifier,
        },
    )

    return {
        "run_name": run_name,
        "run_kind": run_kind,
        "task_id": task_id,
        "suite": suite,
        "seed": seed,
        "trace_source": trace_source,
        "episode_config": engine.episode_config,
        "final_score": round(final_score, 4),
        "cumulative_reward": round(sum(rewards), 4),
        "done": done,
        "steps": len(rewards),
        "grader_breakdown": {key: round(value, 4) for key, value in breakdown.items()},
        "verifier_pass": final_score >= pass_threshold,
        "full_episode_solve": done and final_score >= pass_threshold,
        "deadline_hit_rate": round(float(breakdown.get("deadline_accuracy", 1.0)), 4),
        "privilege_trap_avoidance": 0 if privilege_trap_hits else 1,
        "review_before_act_rate": round(1 - (review_failures / total_steps), 4),
        "invalid_action_rate": round(invalid_actions / total_steps, 4),
        "mean_successful_steps": round(successful_steps / total_steps, 4),
        "unique_action_trace_count": len(set(unique_trace)),
        "trace_integrity": {
            "run_kind": run_kind,
            "exact_scripted_match": bool(actions == SCRIPTED_BASELINES[task_id]) if run_kind == "trace_dir" else False,
        },
        "outcome_correctness": round(float(last_verifier.get("episode_average", {}).get("outcome_correctness", 0.0)), 4),
        "process_compliance": round(float(last_verifier.get("episode_average", {}).get("process_compliance", 0.0)), 4),
        "deadline_or_latency": round(float(last_verifier.get("episode_average", {}).get("deadline_or_latency", 0.0)), 4),
        "safety_or_anti_cheat": round(float(last_verifier.get("episode_average", {}).get("safety_or_anti_cheat", 0.0)), 4),
    }


def aggregate_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate episode metrics by run, suite, and task."""

    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (row["run_name"], row["suite"], row["task_id"])
        grouped.setdefault(key, []).append(row)

    aggregates: List[Dict[str, Any]] = []
    for (run_name, suite, task_id), group in sorted(grouped.items()):
        breakdown_keys = sorted(group[0]["grader_breakdown"].keys())
        _debug_log(
            "Aggregate row computed",
            {
                "run_name": run_name,
                "suite": suite,
                "task_id": task_id,
                "episodes": len(group),
                "keys_present": sorted(group[0].keys()),
            },
        )
        aggregates.append(
            {
                "run_name": run_name,
                "suite": suite,
                "task_id": task_id,
                "episodes": len(group),
                "avg_final_score": round(mean(row["final_score"] for row in group), 4),
                "avg_cumulative_reward": round(mean(row["cumulative_reward"] for row in group), 4),
                "verifier_pass_rate": round(mean(1.0 if row["verifier_pass"] else 0.0 for row in group), 4),
                "full_episode_solve_rate": round(mean(1.0 if row["full_episode_solve"] else 0.0 for row in group), 4),
                "deadline_hit_rate": round(mean(row["deadline_hit_rate"] for row in group), 4),
                "privilege_trap_avoidance_rate": round(mean(row["privilege_trap_avoidance"] for row in group), 4),
                "review_before_act_rate": round(mean(row["review_before_act_rate"] for row in group), 4),
                "invalid_action_rate": round(mean(row["invalid_action_rate"] for row in group), 4),
                "mean_successful_steps": round(mean(row["mean_successful_steps"] for row in group), 4),
                "unique_action_trace_count": sum(row["unique_action_trace_count"] for row in group),
                "outcome_correctness": round(mean(row["outcome_correctness"] for row in group), 4),
                "process_compliance": round(mean(row["process_compliance"] for row in group), 4),
                "deadline_or_latency": round(mean(row["deadline_or_latency"] for row in group), 4),
                "safety_or_anti_cheat": round(mean(row["safety_or_anti_cheat"] for row in group), 4),
                "grader_breakdown": {
                    key: round(mean(row["grader_breakdown"][key] for row in group), 4)
                    for key in breakdown_keys
                },
            }
        )
    return aggregates


def validate_trace_authenticity(rows: List[Dict[str, Any]], run_specs: List[Tuple[str, str, str]]) -> None:
    trace_runs = {name for name, kind, _ in run_specs if kind == "trace_dir"}
    for run_name in sorted(trace_runs):
        run_rows = [row for row in rows if row["run_name"] == run_name]
        if not run_rows:
            continue
        all_scripted = all(bool(row.get("trace_integrity", {}).get("exact_scripted_match", False)) for row in run_rows)
        task_scripted = {
            task_id: all(
                bool(row.get("trace_integrity", {}).get("exact_scripted_match", False))
                for row in run_rows
                if row["task_id"] == task_id
            )
            for task_id in TASK_DEFINITIONS
        }
        _debug_log(
            "Trace authenticity check",
            {
                "run_name": run_name,
                "episodes": len(run_rows),
                "all_exact_scripted_match": all_scripted,
                "task_exact_scripted_match": task_scripted,
            },
        )
        scripted_tasks = sorted([task_id for task_id, scripted in task_scripted.items() if scripted])
        if all_scripted:
            offending_sources = sorted({row.get("trace_source", "") for row in run_rows if row.get("trace_source")})
            raise ValueError(
                f"Run '{run_name}' appears to contain only scripted-baseline-equivalent traces. "
                "This invalidates before/after evidence. Regenerate genuine traces with collect_traces.py "
                "and ensure they differ from SCRIPTED_BASELINES. "
                f"Offending files (sample): {offending_sources[:6]}"
            )
        if scripted_tasks:
            raise ValueError(
                f"Run '{run_name}' contains scripted-equivalent traces for tasks: {scripted_tasks}. "
                "At least one task-level policy file is not genuine model behavior. "
                "Regenerate traces for those tasks with collect_traces.py before judging."
            )


def parse_run_spec(spec: str) -> Tuple[str, str, str]:
    """Parse NAME=KIND[:VALUE]."""

    if "=" not in spec:
        raise ValueError(f"Invalid run spec: {spec}")
    name, payload = spec.split("=", 1)
    if payload == "scripted":
        return name, "scripted", ""
    if payload.startswith("trace_dir:"):
        return name, "trace_dir", payload.split(":", 1)[1]
    raise ValueError(f"Unsupported run spec: {spec}")


def validate_run_specs(run_specs: List[Tuple[str, str, str]], random_seeds: List[int]) -> None:
    for run_name, run_kind, run_value in run_specs:
        if run_kind != "trace_dir":
            continue
        trace_dir = Path(run_value)
        missing = []
        if not trace_dir.exists():
            raise FileNotFoundError(
                f"Trace directory for run '{run_name}' does not exist: {trace_dir}. "
                "Generate traces first with collect_traces.py."
            )
        for task_id in TASK_DEFINITIONS:
            required_pairs: List[Tuple[str, Optional[int]]] = [("fixed", None)] + [("randomized", seed) for seed in random_seeds]
            missing_variants: List[str] = []
            for suite, seed in required_pairs:
                try:
                    resolve_trace_file(trace_dir, task_id, suite, seed)
                except FileNotFoundError:
                    seed_label = "fixed" if seed is None else str(seed)
                    missing_variants.append(f"{suite}:{seed_label}")
            if missing_variants:
                missing.append(task_id)
                _debug_log(
                    "Missing trace variants for task",
                    {
                        "run_name": run_name,
                        "trace_dir": str(trace_dir),
                        "task_id": task_id,
                        "missing_variants": missing_variants,
                    },
                )
        _debug_log(
            "Trace directory preflight",
            {
                "run_name": run_name,
                "trace_dir": str(trace_dir),
                "exists": trace_dir.exists(),
                "missing_tasks": missing,
            },
        )
        if missing:
            raise FileNotFoundError(
                "Missing trace artifacts for run "
                f"'{run_name}' in '{trace_dir}'. Missing task traces: {missing or 'all'}. "
                "Generate traces first, e.g. "
                "python collect_traces.py --run-name base --model-name Qwen/Qwen2.5-1.5B-Instruct "
                "--task-ids task_1 task_2 task_3 --seeds 11 23 37 41 53 "
                "--trace-dir outputs/policies/base --verbose-dir outputs/traces/base --no-scripted-hints"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        help="Run spec: name=scripted (oracle reference) or name=trace_dir:<directory>",
    )
    parser.add_argument(
        "--random-seeds",
        nargs="*",
        type=int,
        default=[11, 23, 37, 41, 53],
        help="Seeds for the randomized suite.",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=0.9,
        help="Final-score threshold used for verifier/full-episode pass rates.",
    )
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    run_specs = [parse_run_spec(spec) for spec in (args.run or ["scripted=scripted"])]
    validate_run_specs(run_specs, args.random_seeds)

    for run_name, run_kind, run_value in run_specs:
        for task_id in TASK_DEFINITIONS:
            rows.append(run_episode(run_name, run_kind, run_value, task_id, "fixed", None, args.pass_threshold))
            for seed in args.random_seeds:
                rows.append(run_episode(run_name, run_kind, run_value, task_id, "randomized", seed, args.pass_threshold))

    validate_trace_authenticity(rows, run_specs)
    aggregates = aggregate_rows(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "episode_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(aggregates, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} episode rows and {len(aggregates)} aggregate rows to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
