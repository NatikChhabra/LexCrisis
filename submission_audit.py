#!/usr/bin/env python3
"""Pre-submission quality gate for LexCrisis hackathon package."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "outputs" / "evals" / "summary.json"
ROWS_PATH = ROOT / "outputs" / "evals" / "episode_rows.json"
TRAINING_PATH = ROOT / "outputs" / "training_metrics.json"
README_PATH = ROOT / "README.md"


class Audit:
    def __init__(self) -> None:
        self.failures: List[str] = []
        self.warnings: List[str] = []
        self.passes: List[str] = []

    def ok(self, message: str) -> None:
        self.passes.append(message)

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_required_files(audit: Audit) -> None:
    required = [
        ROOT / "openenv.yaml",
        ROOT / "main.py",
        ROOT / "Dockerfile",
        ROOT / "requirements.txt",
        README_PATH,
        SUMMARY_PATH,
        ROWS_PATH,
        TRAINING_PATH,
        ROOT / "assets" / "training_loss.png",
        ROOT / "assets" / "reward_curve.png",
        ROOT / "assets" / "score_comparison.png",
    ]
    for path in required:
        if path.exists():
            audit.ok(f"Exists: {path.relative_to(ROOT)}")
        else:
            audit.fail(f"Missing required file: {path.relative_to(ROOT)}")


def _check_readme_links(audit: Audit, readme_text: str) -> None:
    urls = re.findall(r"https?://[^\s)]+", readme_text)
    if not urls:
        audit.fail("README has no links.")
        return
    required_domains = {
        "huggingface.co/spaces": "Hugging Face Space link",
        "colab.research.google.com": "Colab link",
        "github.com": "Repository link",
    }
    for needle, label in required_domains.items():
        if any(needle in url for url in urls):
            audit.ok(f"README includes {label}.")
        else:
            audit.fail(f"README missing {label}.")
    if "recording in progress" in readme_text.lower():
        audit.fail("README demo video is still marked as 'recording in progress'.")
    if "todo_before_submit" in readme_text.lower():
        audit.fail("README still contains TODO placeholder links.")
    if "## Honest Evaluation Pipeline" in readme_text:
        audit.ok("README includes evaluation pipeline section.")
    else:
        audit.fail("README missing explicit evaluation pipeline section.")


def _check_training_metrics(audit: Audit, metrics: Dict[str, Any]) -> None:
    sft_loss = metrics.get("sft_loss")
    if isinstance(sft_loss, list) and len(sft_loss) >= 10:
        audit.ok(f"training_metrics.sft_loss has {len(sft_loss)} points.")
    else:
        audit.fail("training_metrics.sft_loss is missing or too short (<10 points).")


def _check_summary(audit: Audit, summary_rows: List[Dict[str, Any]]) -> None:
    runs = {row.get("run_name") for row in summary_rows}
    for run in ("oracle", "base", "sft"):
        if run in runs:
            audit.ok(f"summary.json contains run: {run}")
        else:
            audit.fail(f"summary.json missing run: {run}")

    # High-signal guard: if base and sft are identical across all task/suite combinations,
    # improvement claims are not credible.
    by_key: Dict[tuple[str, str], Dict[str, float]] = {}
    for row in summary_rows:
        run = row.get("run_name")
        key = (str(row.get("suite")), str(row.get("task_id")))
        by_key.setdefault(key, {})
        by_key[key][run] = float(row.get("avg_final_score", 0.0))

    comparable = [values for values in by_key.values() if "base" in values and "sft" in values]
    if not comparable:
        audit.fail("No base-vs-sft comparable rows found in summary.json.")
        return
    any_delta = any(abs(values["sft"] - values["base"]) > 1e-6 for values in comparable)
    if any_delta:
        audit.ok("At least one base-vs-sft score delta detected.")
    else:
        audit.fail(
            "No measurable base-vs-sft score delta found in summary.json. "
            "Regenerate genuine traces and retrain before submission."
        )


def _check_trace_integrity(audit: Audit, rows: List[Dict[str, Any]]) -> None:
    trace_rows = [r for r in rows if r.get("run_kind") == "trace_dir" and r.get("run_name") in {"base", "sft"}]
    if not trace_rows:
        audit.fail("No trace_dir rows found for base/sft in episode_rows.json.")
        return

    for run_name in ("base", "sft"):
        run_rows = [r for r in trace_rows if r.get("run_name") == run_name]
        if not run_rows:
            audit.fail(f"No rows found for run '{run_name}'.")
            continue
        exact_scripted = [
            bool(r.get("trace_integrity", {}).get("exact_scripted_match", False))
            for r in run_rows
        ]
        if all(exact_scripted):
            audit.fail(
                f"All '{run_name}' trace rows match scripted baseline exactly. "
                "Evidence is not model-distinct."
            )
        else:
            audit.ok(f"Run '{run_name}' includes non-scripted-equivalent traces.")


def main() -> int:
    audit = Audit()
    _check_required_files(audit)

    if README_PATH.exists():
        _check_readme_links(audit, README_PATH.read_text(encoding="utf-8"))
    if TRAINING_PATH.exists():
        _check_training_metrics(audit, _load_json(TRAINING_PATH))
    if SUMMARY_PATH.exists():
        _check_summary(audit, _load_json(SUMMARY_PATH))
    if ROWS_PATH.exists():
        _check_trace_integrity(audit, _load_json(ROWS_PATH))

    print("\n=== Submission Audit ===")
    for message in audit.passes:
        print(f"[PASS] {message}")
    for message in audit.warnings:
        print(f"[WARN] {message}")
    for message in audit.failures:
        print(f"[FAIL] {message}")

    print(
        f"\nResult: {len(audit.passes)} pass, "
        f"{len(audit.warnings)} warn, {len(audit.failures)} fail"
    )
    return 1 if audit.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

