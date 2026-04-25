"""Shared rollout utilities for evaluation, trace collection, and inference."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests as http_requests
from openai import OpenAI

from lexcrisis_env.prompting import SYSTEM_PROMPT

Logger = Optional[Callable[[str], None]]


@dataclass
class GeneratedAction:
    """One model generation and the parsed environment action."""

    raw_text: str
    action: Dict[str, Any]
    repaired: bool = False


def build_user_message(task_id: str, step_index: int, observation: Dict[str, Any]) -> str:
    """Build a judge-safe prompt from the current observation only."""

    context = {
        "task_id": task_id,
        "task_name": observation.get("task_name", task_id),
        "difficulty": observation.get("difficulty"),
        "step": step_index,
        "steps_remaining": max(0, observation.get("max_steps", 0) - step_index),
        "task_description": observation.get("task_description", ""),
        "available_actions": observation.get("available_actions", []),
        "findings_so_far": {k: v for k, v in observation.get("findings", {}).items() if v},
        "active_deadlines": observation.get("active_deadlines", []),
        "ethical_alerts": observation.get("ethical_alerts", []),
        "episode_config": observation.get("episode_config", {}),
        "feedback_from_last_action": observation.get("feedback", ""),
        "selectable_items": observation.get("documents", []),
    }
    if observation.get("current_content"):
        context["revealed_content"] = observation["current_content"]

    lines = [
        "Choose the single best next environment action.",
        "If the necessary evidence has not been revealed yet, prefer a review action instead of guessing.",
        "Use only the action names in available_actions and only the IDs present in the observation.",
        json.dumps(context, indent=2),
        "",
        'Return exactly one JSON object with keys "action_type" and "parameters".',
    ]
    return "\n".join(lines)


def build_repair_message(available_actions: List[str]) -> str:
    """Repair instruction used after an unparsable model response."""

    return (
        "The previous answer was not a valid environment action. "
        f"Return exactly one JSON object using one of these action types: {available_actions}. "
        'Format: {"action_type": "<type>", "parameters": {...}}'
    )


def parse_action_payload(text: str, available_actions: List[str]) -> Optional[Dict[str, Any]]:
    """Extract the first valid action object from model output."""

    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    candidates: List[str] = []
    last_brace = cleaned.rfind("}")
    first_brace = cleaned.find("{")
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        candidates.append(cleaned[first_brace : last_brace + 1])
    candidates.extend(re.findall(r"\{[^{}]*\}", cleaned, re.DOTALL))
    if cleaned:
        candidates.append(cleaned)

    seen = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        action_type = obj.get("action_type")
        if action_type in available_actions and isinstance(obj.get("parameters", {}), dict):
            return {"action_type": action_type, "parameters": obj.get("parameters", {})}
    return None


class RemoteChatBackend:
    """OpenAI-compatible remote chat backend for Hugging Face router usage."""

    def __init__(
        self,
        *,
        model_name: str,
        api_base_url: Optional[str] = None,
        hf_token: Optional[str] = None,
        timeout: float = 120.0,
        logger: Logger = None,
    ) -> None:
        self.model_name = model_name.strip()
        self.api_base_url = (api_base_url or os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")).strip().rstrip("/")
        self.hf_token = (hf_token or os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "").strip()
        self.timeout = timeout
        self.logger = logger
        self.client = OpenAI(
            base_url=self.api_base_url,
            api_key=self.hf_token,
            timeout=timeout,
            max_retries=3,
        )

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(message)

    def complete(self, messages: List[Dict[str, str]], max_tokens: int = 256) -> str:
        """Generate a response from the configured remote model."""

        if not self.hf_token:
            raise RuntimeError("HF_TOKEN or API_KEY must be set for remote trace collection.")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0,
                max_tokens=max_tokens,
                messages=messages,
            )
            return response.choices[0].message.content or ""
        except Exception as sdk_exc:
            self._log(f"SDK generation failed: {sdk_exc}")
            response = http_requests.post(
                f"{self.api_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.hf_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": max_tokens,
                },
                timeout=self.timeout,
            )
            if not response.ok:
                raise RuntimeError(
                    f"Remote generation failed with HTTP {response.status_code}: {response.text[:300]}"
                ) from sdk_exc
            payload = response.json()
            return payload.get("choices", [{}])[0].get("message", {}).get("content", "")


class LocalChatBackend:
    """Local transformers or PEFT-backed chat model."""

    def __init__(self, *, model_path: str, logger: Logger = None) -> None:
        self.model_path = str(model_path)
        self.logger = logger
        self.tokenizer, self.model = self._load_model(Path(model_path))

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(message)

    def _load_model(self, model_path: Path) -> Any:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local model trace collection requires transformers and torch. "
                "Install them in the training environment before using --model-path."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if (model_path / "adapter_config.json").exists():
            try:
                from peft import PeftConfig, PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "This checkpoint looks like a PEFT adapter. Install peft to load it."
                ) from exc
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

        model.eval()
        return tokenizer, model

    def complete(self, messages: List[Dict[str, str]], max_tokens: int = 256) -> str:
        """Generate a response from the local model."""

        import torch

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n\n".join(f"{item['role'].upper()}: {item['content']}" for item in messages)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def make_backend(
    *,
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    api_base_url: Optional[str] = None,
    hf_token: Optional[str] = None,
    logger: Logger = None,
) -> Any:
    """Build the appropriate chat backend from CLI args."""

    if bool(model_name) == bool(model_path):
        raise ValueError("Provide exactly one of model_name or model_path.")
    if model_path:
        return LocalChatBackend(model_path=model_path, logger=logger)
    return RemoteChatBackend(
        model_name=model_name or "",
        api_base_url=api_base_url,
        hf_token=hf_token,
        logger=logger,
    )


class PolicyAgent:
    """Thin wrapper that prompts the model and repairs invalid JSON once."""

    def __init__(
        self,
        backend: Any,
        *,
        history_window: int = 8,
        repair_attempts: int = 1,
        logger: Logger = None,
    ) -> None:
        self.backend = backend
        self.history_window = history_window
        self.repair_attempts = repair_attempts
        self.logger = logger

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(message)

    def generate_action(
        self,
        *,
        task_id: str,
        step_index: int,
        observation: Dict[str, Any],
        history: List[Dict[str, str]],
    ) -> GeneratedAction:
        """Generate one parseable action for the current observation."""

        available_actions = observation.get("available_actions", [])
        messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history[-self.history_window :])
        messages.append({"role": "user", "content": build_user_message(task_id, step_index, observation)})

        raw = self.backend.complete(messages)
        parsed = parse_action_payload(raw, available_actions)
        if parsed is not None:
            return GeneratedAction(raw_text=raw, action=parsed, repaired=False)

        repair_messages = list(messages)
        repair_messages.append({"role": "assistant", "content": raw})
        repair_messages.append({"role": "user", "content": build_repair_message(available_actions)})
        for attempt in range(self.repair_attempts):
            repaired_raw = self.backend.complete(repair_messages)
            repaired = parse_action_payload(repaired_raw, available_actions)
            if repaired is not None:
                return GeneratedAction(raw_text=repaired_raw, action=repaired, repaired=True)
            self._log(f"Repair attempt {attempt + 1} failed for step {step_index}.")
            repair_messages[-2] = {"role": "assistant", "content": repaired_raw}

        raise ValueError(f"Model output was not a valid action.\nRaw output:\n{raw}")
