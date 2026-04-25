#!/usr/bin/env python3
"""Canonical LexCrisis supervised fine-tuning runner."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load newline-delimited JSON records from disk."""

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No training rows found in {path}")
    return rows


def conversations_to_text(example: Dict[str, Any], tokenizer: Any) -> str:
    """Render ShareGPT-style conversations with the tokenizer chat template."""

    role_map = {
        "system": "system",
        "human": "user",
        "gpt": "assistant",
    }
    messages = [
        {
            "role": role_map.get(turn["from"], "user"),
            "content": turn["value"],
        }
        for turn in example["conversations"]
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return "\n\n".join(f"{item['role'].upper()}: {item['content']}" for item in messages)


def load_model_and_tokenizer(args: argparse.Namespace) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load the SFT model stack, preferring Unsloth and falling back to HF PEFT."""

    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
    except ImportError:
        FastLanguageModel = None
        is_bfloat16_supported = None

    if FastLanguageModel is not None:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=TARGET_MODULES,
            lora_alpha=args.lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )
        info = {
            "backend": "unsloth",
            "bf16": bool(is_bfloat16_supported()),
            "fp16": not bool(is_bfloat16_supported()),
        }
        return model, tokenizer, info

    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "train_sft.py requires either Unsloth or the transformers/peft/torch stack. "
            "Install the Colab training dependencies before running it."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=TARGET_MODULES,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    info = {
        "backend": "transformers",
        "bf16": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "fp16": not bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    }
    return model, tokenizer, info


def build_trainer(
    *,
    model: Any,
    tokenizer: Any,
    dataset: Any,
    args: argparse.Namespace,
    backend_info: Dict[str, Any],
) -> Any:
    """Construct a TRL SFTTrainer without depending on one specific minor version."""

    from trl import SFTConfig, SFTTrainer

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to="none",
        lr_scheduler_type="linear",
        weight_decay=0.01,
        optim="adamw_8bit" if backend_info["backend"] == "unsloth" else "adamw_torch",
        seed=args.seed,
        # Keep only model-relevant columns during collation to avoid
        # string metadata (task_id/action_type) tensorization crashes.
        remove_unused_columns=True,
        fp16=backend_info["fp16"],
        bf16=backend_info["bf16"],
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
    )

    trainer_signature = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "train_dataset": dataset,
        "args": training_args,
    }
    if "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    return SFTTrainer(**trainer_kwargs)


def update_metrics_file(path: Path, payload: Dict[str, Any]) -> None:
    """Merge the latest SFT metrics into the shared metrics file."""

    existing: Dict[str, Any] = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.update(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Base model name, for example unsloth/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data", type=Path, required=True, help="Path to outputs/self_improve/sft_examples.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU-only training (very slow, not recommended for competition runs).",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        torch = None

    has_cuda = bool(torch is not None and torch.cuda.is_available())
    if not has_cuda and not args.allow_cpu:
        raise RuntimeError(
            "No CUDA GPU detected. SFT on CPU is extremely slow and appears stalled at 0%% for long periods.\n"
            "Best action for hackathon quality and speed: run on Colab/HF GPU and rerun this exact command there.\n"
            "If you still want a local smoke test, append --allow-cpu --max-steps 5 --max-seq-length 512."
        )

    rows = load_jsonl(args.data)
    model, tokenizer, backend_info = load_model_and_tokenizer(args)
    # Right-padding is recommended for half precision SFT in TRL.
    tokenizer.padding_side = "right"

    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("train_sft.py requires the datasets library.") from exc

    formatted_rows = []
    for row in rows:
        formatted_rows.append({"text": conversations_to_text(row, tokenizer)})
    dataset = Dataset.from_list(formatted_rows)
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        args=args,
        backend_info=backend_info,
    )
    train_result = trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    log_history = getattr(trainer.state, "log_history", [])
    sft_loss = [round(float(entry["loss"]), 4) for entry in log_history if "loss" in entry]
    metrics_payload = {
        "model_name": args.model,
        "sft_backend": backend_info["backend"],
        "sft_data_path": str(args.data),
        "sft_output_dir": str(args.output_dir),
        "sft_examples": len(formatted_rows),
        "sft_loss": sft_loss,
        "sft_train_runtime_seconds": round(float(getattr(train_result, "metrics", {}).get("train_runtime", 0.0)), 4),
        "sft_train_steps_per_second": round(float(getattr(train_result, "metrics", {}).get("train_steps_per_second", 0.0)), 4),
    }
    update_metrics_file(args.metrics, metrics_payload)
    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
