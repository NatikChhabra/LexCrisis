#!/usr/bin/env python3
"""Build a concise, judge-facing markdown report from evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "outputs" / "evals" / "summary.json"
ROWS_PATH = ROOT / "outputs" / "evals" / "episode_rows.json"
OUT_PATH = ROOT / "outputs" / "evals" / "judge_report.md"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _index_summary(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    return {
        (str(row["run_name"]), str(row["suite"]), str(row["task_id"])): row
        for row in rows
    }


def _format_delta(value: float) -> str:
    return f"{value:+.4f}"


def _best_examples(episode_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_task: Dict[str, Dict[str, Any]] = {}
    for task_id in ("task_1", "task_2", "task_3"):
        base_candidates = [
            row for row in episode_rows
            if row.get("run_name") == "base" and row.get("task_id") == task_id
        ]
        sft_candidates = [
            row for row in episode_rows
            if row.get("run_name") == "sft" and row.get("task_id") == task_id
        ]
        best = None
        for base_row in base_candidates:
            for sft_row in sft_candidates:
                if base_row.get("suite") != sft_row.get("suite"):
                    continue
                if base_row.get("seed") != sft_row.get("seed"):
                    continue
                delta = float(sft_row.get("final_score", 0.0)) - float(base_row.get("final_score", 0.0))
                sample = {
                    "suite": base_row.get("suite"),
                    "seed": base_row.get("seed"),
                    "base_score": float(base_row.get("final_score", 0.0)),
                    "sft_score": float(sft_row.get("final_score", 0.0)),
                    "delta": delta,
                }
                if best is None or delta > best["delta"]:
                    best = sample
        if best is not None:
            by_task[task_id] = best
    return by_task


def main() -> int:
    if not SUMMARY_PATH.exists() or not ROWS_PATH.exists():
        raise FileNotFoundError("Missing evaluation artifacts. Run evaluate_runs.py first.")

    summary_rows: List[Dict[str, Any]] = _load_json(SUMMARY_PATH)
    episode_rows: List[Dict[str, Any]] = _load_json(ROWS_PATH)
    idx = _index_summary(summary_rows)

    lines: List[str] = []
    lines.append("# LexCrisis Judge Report")
    lines.append("")
    lines.append("Auto-generated from `outputs/evals/summary.json` and `outputs/evals/episode_rows.json`.")
    lines.append("")
    lines.append("## Base vs SFT Aggregate Deltas")
    lines.append("")
    lines.append("| Suite | Task | Base score | SFT score | Delta (SFT-Base) |")
    lines.append("| --- | --- | ---: | ---: | ---: |")

    any_positive_delta = False
    for suite in ("fixed", "randomized"):
        for task_id in ("task_1", "task_2", "task_3"):
            base = idx.get(("base", suite, task_id))
            sft = idx.get(("sft", suite, task_id))
            if not base or not sft:
                continue
            base_score = float(base.get("avg_final_score", 0.0))
            sft_score = float(sft.get("avg_final_score", 0.0))
            delta = sft_score - base_score
            if delta > 1e-6:
                any_positive_delta = True
            lines.append(
                f"| {suite} | {task_id} | {base_score:.4f} | {sft_score:.4f} | {_format_delta(delta)} |"
            )

    lines.append("")
    lines.append("## Representative Episode Comparisons")
    lines.append("")

    best = _best_examples(episode_rows)
    for task_id in ("task_1", "task_2", "task_3"):
        item = best.get(task_id)
        if not item:
            lines.append(f"- {task_id}: no matched base/sft episode pairs found.")
            continue
        seed_label = "fixed" if item["seed"] is None else str(item["seed"])
        lines.append(
            f"- {task_id} ({item['suite']} / seed={seed_label}): "
            f"base={item['base_score']:.4f}, sft={item['sft_score']:.4f}, delta={_format_delta(item['delta'])}"
        )

    lines.append("")
    lines.append("## Readiness Signal")
    lines.append("")
    if any_positive_delta:
        lines.append("- Improvement evidence detected in aggregate deltas.")
    else:
        lines.append(
            "- No positive base-vs-sft aggregate deltas detected. "
            "Do not claim training improvement until traces are regenerated."
        )

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote judge report to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

