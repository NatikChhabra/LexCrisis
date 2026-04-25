#!/usr/bin/env python3
"""Collect honest LexCrisis rollouts and write policy/tracing artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import os

from lexcrisis_env.env import LexCrisisEngine
from lexcrisis_env.models import Action
from lexcrisis_env.policy_runtime import PolicyAgent, make_backend
from lexcrisis_env.tasks import TASK_DEFINITIONS

def _log(message: str) -> None:
    sys.stderr.write(f"# {message}\n")
    sys.stderr.flush()


def _seed_label(seed: Optional[int]) -> str:
    return "fixed" if seed is None else str(seed)


def _policy_filename(task_id: str, suite: str, seed: Optional[int]) -> str:
    if suite == "fixed":
        return f"{task_id}__fixed.json"
    return f"{task_id}__randomized__{_seed_label(seed)}.json"


def _trace_stem(task_id: str, suite: str, seed: Optional[int]) -> str:
    return f"{task_id}__{suite}__{_seed_label(seed)}"


def _relevant_item_id(action: Dict[str, Any], observation: Dict[str, Any]) -> str:
    params = action.get("parameters", {})
    for key in ("client_id", "doc_id", "event_id", "item_id", "request_id", "expert_id"):
        if params.get(key):
            return str(params[key])
    if observation.get("documents"):
        return str(observation["documents"][0].get("item_id", ""))
    return ""


def _render_trace_markdown(trace_rows: List[Dict[str, Any]]) -> str:
    if not trace_rows:
        return "# Empty trace\n"

    header = trace_rows[0]
    lines = [
        f"# {header['run_name']} - {header['task_id']} - {header['suite']} - {header['seed_label']}",
        "",
        f"- Final score: `{header.get('final_score', 0.0):.4f}`",
        f"- Completed: `{header.get('done', False)}`",
        f"- Episode config: `{json.dumps(header.get('episode_config', {}), sort_keys=True)}`",
        "",
        "| Step | Revealed Item | Action | Feedback | Reward | Verifier Signals | Final Score |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in trace_rows:
        action_text = json.dumps(row.get("action", {}), sort_keys=True)
        verifier = json.dumps(row.get("verifier_signals", {}), sort_keys=True)
        feedback = str(row.get("feedback", "")).replace("\n", " ").replace("|", "/")
        lines.append(
            f"| {row.get('step', '')} | {row.get('revealed_item', '')} | `{action_text}` | "
            f"{feedback} | `{row.get('reward', 0.0):.4f}` | `{verifier}` | `{row.get('final_score', 0.0):.4f}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def collect_episode(
    *,
    agent: PolicyAgent,
    run_name: str,
    task_id: str,
    suite: str,
    seed: Optional[int],
    trace_dir: Path,
    verbose_dir: Path,
) -> Dict[str, Any]:
    """Run one episode and persist both compact and verbose artifacts."""

    engine = LexCrisisEngine()
    observation = engine.reset(task_id=task_id, seed=seed)
    history: List[Dict[str, str]] = []
    executed_actions: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    step_index = 0
    done = False
    had_error = False

    while not done and step_index < observation.max_steps:
        step_index += 1
        obs_dict = observation.model_dump(mode="json")
        try:
            generated = agent.generate_action(
                task_id=task_id,
                step_index=step_index,
                observation=obs_dict,
                history=history,
            )
            action_model = Action.model_validate(generated.action)
            action_dict = action_model.model_dump(mode="json")
            next_observation, reward, done, info = engine.step(action_model)
            trace_rows.append(
                {
                    "run_name": run_name,
                    "task_id": task_id,
                    "suite": suite,
                    "seed": seed,
                    "seed_label": _seed_label(seed),
                    "step": step_index,
                    "episode_config": engine.episode_config,
                    "observation": obs_dict,
                    "raw_model_output": generated.raw_text,
                    "repaired_output": generated.repaired,
                    "action": action_dict,
                    "revealed_item": _relevant_item_id(action_dict, obs_dict),
                    "feedback": next_observation.feedback,
                    "reward": round(float(reward), 4),
                    "done": done,
                    "score": round(float(info.get("score", engine.last_score)), 4),
                    "reward_breakdown": info.get("reward_breakdown", {}),
                    "verifier_signals": info.get("reward_breakdown", {}).get("verifier_signals", {}),
                    "current_content_after_action": next_observation.current_content,
                }
            )
            executed_actions.append(action_dict)
            history.append({"role": "assistant", "content": json.dumps(action_dict, sort_keys=True)})
            history.append(
                {
                    "role": "user",
                    "content": (
                        f"[ENV step={step_index}] reward={float(reward):.4f} "
                        f"score={engine.last_score:.4f} | {next_observation.feedback}"
                    ),
                }
            )
            observation = next_observation
        except Exception as exc:
            had_error = True
            trace_rows.append(
                {
                    "run_name": run_name,
                    "task_id": task_id,
                    "suite": suite,
                    "seed": seed,
                    "seed_label": _seed_label(seed),
                    "step": step_index,
                    "episode_config": engine.episode_config,
                    "observation": obs_dict,
                    "action": {},
                    "revealed_item": "",
                    "feedback": f"Generation or environment error: {exc}",
                    "reward": 0.0,
                    "done": False,
                    "score": round(engine.last_score, 4),
                    "reward_breakdown": {},
                    "verifier_signals": {},
                    "error": str(exc),
                }
            )
            _log(f"{run_name}/{task_id}/{suite}/{_seed_label(seed)} failed at step {step_index}: {exc}")
            break

    final_score = round(engine.last_score, 4)
    for row in trace_rows:
        row["final_score"] = final_score

    trace_dir.mkdir(parents=True, exist_ok=True)
    verbose_dir.mkdir(parents=True, exist_ok=True)
    policy_path = trace_dir / _policy_filename(task_id, suite, seed)
    trace_path = verbose_dir / f"{_trace_stem(task_id, suite, seed)}.jsonl"
    markdown_path = verbose_dir / f"{_trace_stem(task_id, suite, seed)}.md"
    policy_path.write_text(json.dumps(executed_actions, indent=2), encoding="utf-8")
    _write_jsonl(trace_path, trace_rows)
    markdown_path.write_text(_render_trace_markdown(trace_rows), encoding="utf-8")

    _log(
        f"Wrote {policy_path.name}, {trace_path.name}, and {markdown_path.name} "
        f"(steps={len(executed_actions)}, score={final_score:.4f})"
    )
    return {
        "task_id": task_id,
        "suite": suite,
        "seed": seed,
        "steps": len(executed_actions),
        "final_score": final_score,
        "had_error": had_error,
        "policy_path": str(policy_path),
        "trace_path": str(trace_path),
        "markdown_path": str(markdown_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True, choices=["base", "sft", "rl"])
    parser.add_argument("--model-name", default=None, help="Remote model name for router-based collection.")
    parser.add_argument("--model-path", default=None, help="Local model or adapter path for collection.")
    parser.add_argument("--api-base-url", default=None, help="Optional OpenAI-compatible API base URL.")
    parser.add_argument("--hf-token", default=None, help="Optional API token override for remote collection.")
    parser.add_argument(
        "--task-ids",
        nargs="+",
        choices=list(TASK_DEFINITIONS.keys()),
        default=list(TASK_DEFINITIONS.keys()),
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 23, 37, 41, 53])
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--verbose-dir", type=Path, required=True)
    parser.add_argument("--history-window", type=int, default=8)
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument(
        "--no-scripted-hints",
        action="store_true",
        default=False,
        help="Kept for compatibility. Trace collection never injects oracle hints.",
    )
    args = parser.parse_args()
    env_hf = bool(os.getenv("HF_TOKEN"))
    env_api = bool(os.getenv("API_KEY"))
    _log(
        "Trace collection config: "
        f"run={args.run_name} model_name_set={bool(args.model_name)} model_path_set={bool(args.model_path)} "
        f"api_base_url_set={bool(args.api_base_url)} token_cli_set={bool(args.hf_token)} "
        f"token_env_set={env_hf or env_api} tasks={len(args.task_ids)} seeds={len(args.seeds)}"
    )
    if args.model_name and not (args.hf_token or env_hf or env_api):
        raise RuntimeError(
            "Remote trace collection requires HF_TOKEN or API_KEY. "
            "Set one of these env vars (or pass --hf-token) before running collect_traces.py."
        )

    try:
        backend = make_backend(
            model_name=args.model_name,
            model_path=args.model_path,
            api_base_url=args.api_base_url,
            hf_token=args.hf_token,
            logger=_log,
        )
    except Exception as exc:
        raise
    agent = PolicyAgent(
        backend,
        history_window=args.history_window,
        repair_attempts=args.repair_attempts,
        logger=_log,
    )

    summaries: List[Dict[str, Any]] = []
    for task_id in args.task_ids:
        summaries.append(
            collect_episode(
                agent=agent,
                run_name=args.run_name,
                task_id=task_id,
                suite="fixed",
                seed=None,
                trace_dir=args.trace_dir,
                verbose_dir=args.verbose_dir,
            )
        )
        for seed in args.seeds:
            summaries.append(
                collect_episode(
                    agent=agent,
                    run_name=args.run_name,
                    task_id=task_id,
                    suite="randomized",
                    seed=seed,
                    trace_dir=args.trace_dir,
                    verbose_dir=args.verbose_dir,
                )
            )

    summary_path = args.verbose_dir / "collection_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    _log(f"Wrote collection summary to {summary_path}")
    error_episodes = [entry for entry in summaries if entry.get("had_error")]
    if error_episodes:
        failed_examples = ", ".join(
            f"{entry['task_id']}/{entry['suite']}/{_seed_label(entry['seed'])}"
            for entry in error_episodes[:5]
        )
        raise RuntimeError(
            "Trace collection completed with episode errors. "
            f"Failed episodes: {len(error_episodes)}/{len(summaries)}. "
            f"Examples: {failed_examples}. "
            "Check HF_TOKEN/API_KEY or model-path configuration before evaluating traces."
        )


if __name__ == "__main__":
    main()
