#!/usr/bin/env python3
"""Experimental short-step GRPO refinement for LexCrisis."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lexcrisis_env.env import LexCrisisEngine
from lexcrisis_env.models import Action
from lexcrisis_env.policy_runtime import build_user_message, parse_action_payload
from lexcrisis_env.tasks import SCRIPTED_BASELINES, TASK_DEFINITIONS


def completion_to_text(completion: Any) -> str:
    """Normalise TRL completion payloads into raw text."""

    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        content = completion.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        return json.dumps(completion)
    if isinstance(completion, list):
        return "".join(completion_to_text(item) for item in completion)
    return str(completion)


def build_step_dataset(task_ids: List[str], seeds: List[int]) -> List[Dict[str, Any]]:
    """Create prompt states by replaying the oracle reference trajectories."""

    rows: List[Dict[str, Any]] = []
    for task_id in task_ids:
        for seed in [None] + seeds:
            engine = LexCrisisEngine()
            observation = engine.reset(task_id=task_id, seed=seed)
            prior_actions: List[Dict[str, Any]] = []
            step_index = 0
            done = False
            while not done and step_index < len(SCRIPTED_BASELINES[task_id]):
                step_index += 1
                obs_dict = observation.model_dump(mode="json")
                rows.append(
                    {
                        "prompt": build_user_message(task_id, step_index, obs_dict),
                        "task_id": task_id,
                        "seed": -1 if seed is None else seed,
                        "prior_actions_json": json.dumps(prior_actions),
                    }
                )
                action_model = Action.model_validate(SCRIPTED_BASELINES[task_id][step_index - 1])
                prior_actions.append(action_model.model_dump(mode="json"))
                observation, _, done, _ = engine.step(action_model)
    return rows


def load_local_model(model_path: Path) -> Tuple[Any, Any]:
    """Load a local PEFT adapter or merged model."""

    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("train_grpo.py requires transformers and torch.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if (model_path / "adapter_config.json").exists():
        try:
            from peft import PeftConfig, PeftModel
        except ImportError as exc:
            raise RuntimeError("train_grpo.py requires peft to load adapter checkpoints.") from exc
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
        )
    model.train()
    return tokenizer, model


def merge_metrics_file(path: Path, payload: Dict[str, Any]) -> None:
    """Merge GRPO metrics into the shared metrics file."""

    existing: Dict[str, Any] = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.update(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--task-ids", nargs="+", choices=list(TASK_DEFINITIONS.keys()), default=["task_1", "task_3"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--max-train-steps", type=int, default=200)
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 23, 37, 41, 53])
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("train_grpo.py requires datasets and trl.") from exc

    dataset_rows = build_step_dataset(args.task_ids, args.seeds)
    dataset = Dataset.from_list(dataset_rows)
    tokenizer, model = load_local_model(args.model_path)

    def reward_fn(completions: List[Any], task_id: List[str], seed: List[int], prior_actions_json: List[str], **_: Any) -> List[float]:
        rewards: List[float] = []
        for completion, task_name, seed_value, prior_json in zip(completions, task_id, seed, prior_actions_json):
            raw_text = completion_to_text(completion)
            replay_seed: Optional[int] = None if int(seed_value) < 0 else int(seed_value)
            engine = LexCrisisEngine()
            observation = engine.reset(task_id=task_name, seed=replay_seed)
            try:
                prior_actions = json.loads(prior_json)
                for raw_action in prior_actions:
                    observation, _, done, _ = engine.step(Action.model_validate(raw_action))
                    if done:
                        break
                parsed = parse_action_payload(raw_text, observation.available_actions)
                if parsed is None:
                    rewards.append(-0.05)
                    continue
                _, reward, _, info = engine.step(Action.model_validate(parsed))
                signals = info.get("reward_breakdown", {}).get("verifier_signals", {})
                shaped_reward = float(reward) + (
                    0.02 * float(signals.get("process_compliance", 0.0))
                    + 0.02 * float(signals.get("safety_or_anti_cheat", 0.0))
                    + 0.01 * float(signals.get("deadline_or_latency", 0.0))
                )
                rewards.append(round(shaped_reward, 4))
            except Exception:
                rewards.append(-0.05)
        return rewards

    config_signature = inspect.signature(GRPOConfig.__init__).parameters
    candidate_config: Dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "learning_rate": args.learning_rate,
        "max_steps": args.max_train_steps,
        "logging_steps": 5,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "seed": args.seed,
        "report_to": [],
    }
    optional_config = {
        "num_generations": 4,
        "max_prompt_length": 2048,
        "max_completion_length": 128,
    }
    config_kwargs = {key: value for key, value in candidate_config.items() if key in config_signature}
    for key, value in optional_config.items():
        if key in config_signature:
            config_kwargs[key] = value
    config = GRPOConfig(**config_kwargs)

    trainer_signature = inspect.signature(GRPOTrainer.__init__).parameters
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": config,
        "train_dataset": dataset,
    }
    if "reward_funcs" in trainer_signature:
        trainer_kwargs["reward_funcs"] = reward_fn
    elif "reward_func" in trainer_signature:
        trainer_kwargs["reward_func"] = reward_fn
    if "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    trainer = GRPOTrainer(**trainer_kwargs)
    train_result = trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    log_history = getattr(trainer.state, "log_history", [])
    reward_curve = [
        round(float(entry["reward"]), 4)
        for entry in log_history
        if "reward" in entry
    ]
    payload = {
        "grpo_output_dir": str(args.output_dir),
        "grpo_task_ids": args.task_ids,
        "grpo_dataset_rows": len(dataset_rows),
        "grpo_mean_reward": reward_curve,
        "grpo_train_runtime_seconds": round(float(getattr(train_result, "metrics", {}).get("train_runtime", 0.0)), 4),
    }
    merge_metrics_file(args.metrics, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
