#!/usr/bin/env python3
"""Run honest LexCrisis rollouts against a remote or local policy model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from lexcrisis_env.env import BENCHMARK_NAME, LexCrisisEngine
from lexcrisis_env.models import Action
from lexcrisis_env.policy_runtime import PolicyAgent, make_backend
from lexcrisis_env.tasks import TASK_DEFINITIONS

# ── Domain-Specific Hazard Constants ──────────────────────────────────────────
# These are referenced in the system prompt (lexcrisis_env/prompting.py) and
# directly affect grader scoring:
#
# BCI Rule 33 — conflict of interest prohibition
# BSA Section 132 - attorney-client privilege shield
# BSA Section 134 - litigation-preparation work-product protection
#
# The privilege_trap (respond_discovery with response_type="produce" and
# no objections) triggers an irreversible -0.12 penalty and ethical alert.
# Agents must flag_adversarial before responding to discovery demands.
#
# The current_content field in the observation is populated only after a
# review action — it is the core partial-observability mechanism.

# Default generation budget for structured JSON action output
# Usage: backend.complete(messages, max_tokens=512)
MAX_TOKENS = 512


def _log(message: str) -> None:
    sys.stderr.write(f"# {message}\n")
    sys.stderr.flush()


def _clamp_emit(value: float) -> float:
    return round(max(0.001, min(float(value), 0.999)), 4)


def emit_start(task_id: str, model_label: str) -> None:
    print(f"[START] task={task_id} env={BENCHMARK_NAME} model={model_label}")
    sys.stdout.flush()


def emit_step(step: int, action: Dict[str, Any], reward: float, done: bool, error: Optional[str]) -> None:
    error_value = error if error is not None else "null"
    print(
        f"[STEP] step={step} action={json.dumps(action, sort_keys=True, separators=(',', ':'))} "
        f"reward={_clamp_emit(reward):.4f} done={str(done).lower()} error={error_value}"
    )
    sys.stdout.flush()


def emit_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_text = ",".join(f"{_clamp_emit(reward):.4f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={_clamp_emit(score):.4f} rewards={rewards_text}"
    )
    sys.stdout.flush()


def run_task(
    *,
    agent: PolicyAgent,
    task_id: str,
    seed: Optional[int],
    model_label: str,
) -> Dict[str, Any]:
    """Run one task until the environment ends or the model errors."""

    engine = LexCrisisEngine()
    observation = engine.reset(task_id=task_id, seed=seed)
    history: List[Dict[str, str]] = []
    rewards: List[float] = []
    step_index = 0
    success = False

    emit_start(task_id, model_label)
    while not success and step_index < observation.max_steps:
        step_index += 1
        obs_dict = observation.model_dump(mode="json")
        error_message: Optional[str] = None
        done = False
        reward = 0.0
        action_dict: Dict[str, Any] = {}
        try:
            generated = agent.generate_action(
                task_id=task_id,
                step_index=step_index,
                observation=obs_dict,
                history=history,
            )
            action_model = Action.model_validate(generated.action)
            action_dict = action_model.model_dump(mode="json")
            observation, reward, done, info = engine.step(action_model)
            history.append({"role": "assistant", "content": json.dumps(action_dict, sort_keys=True)})
            history.append(
                {
                    "role": "user",
                    "content": (
                        f"[ENV step={step_index}] reward={float(reward):.4f} "
                        f"score={info.get('score', engine.last_score):.4f} | {observation.feedback}"
                    ),
                }
            )
            success = done
        except Exception as exc:
            error_message = str(exc)
            _log(f"Task {task_id} failed at step {step_index}: {error_message}")

        rewards.append(_clamp_emit(reward))
        emit_step(step_index, action_dict, reward, done, error_message)
        if error_message is not None:
            break

    final_score = _clamp_emit(engine.last_score)
    emit_end(success, step_index, final_score, rewards)
    return {
        "task_id": task_id,
        "task_name": TASK_DEFINITIONS[task_id].name,
        "score": final_score,
        "steps": step_index,
        "success": success,
        "rewards": rewards,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Remote model name. Defaults to the model the README and Blog report.",
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--task-ids", nargs="+", choices=list(TASK_DEFINITIONS.keys()), default=list(TASK_DEFINITIONS.keys()))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--history-window", type=int, default=8)
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("outputs") / "inference_scores.json")
    args = parser.parse_args()

    backend = make_backend(
        model_name=None if args.model_path else args.model_name,
        model_path=args.model_path,
        api_base_url=args.api_base_url,
        hf_token=args.hf_token,
        logger=_log,
    )
    model_label = args.model_path or args.model_name
    agent = PolicyAgent(
        backend,
        history_window=args.history_window,
        repair_attempts=args.repair_attempts,
        logger=_log,
    )

    _log(f"Starting LexCrisis inference runner with model={model_label}")
    results = [
        run_task(
            agent=agent,
            task_id=task_id,
            seed=args.seed,
            model_label=model_label,
        )
        for task_id in args.task_ids
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _log(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
