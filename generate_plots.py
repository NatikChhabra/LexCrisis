#!/usr/bin/env python3
"""Generate evaluation plots from reproducible artifact files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("assets")
DEFAULT_SUMMARY = Path("outputs") / "evals" / "summary.json"
DEFAULT_ROWS = Path("outputs") / "evals" / "episode_rows.json"
DEFAULT_TRAINING = Path("outputs") / "training_metrics.json"

BG = "#111315"
AX = "#1A1D21"
GRID = "#3C4048"
TEXT = "#F5F3EF"
CYAN = "#4DB6E5"
GREEN = "#7CCB92"
AMBER = "#E7B75C"
CORAL = "#E37A6F"
RUN_LABELS = {
    "oracle": "Oracle Reference",
    "base": "Base Model",
    "sft": "SFT Model",
    "rl": "GRPO Model",
    "degraded": "Degraded Policy",
    "scripted_oracle": "Scripted Oracle",
    "curriculum": "Curriculum Variant",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.facecolor": AX,
        "figure.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "text.color": TEXT,
        "grid.color": GRID,
        "grid.alpha": 0.2,
        "legend.facecolor": AX,
        "legend.edgecolor": GRID,
    }
)


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def plot_training_loss(training_metrics: Dict[str, Any]) -> None:
    losses = training_metrics.get("sft_loss")
    rl_rewards = training_metrics.get("grpo_mean_reward")
    if (not isinstance(losses, list) or not losses) and (not isinstance(rl_rewards, list) or not rl_rewards):
        raise ValueError("training_metrics.json must contain sft_loss and/or grpo_mean_reward data.")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    legends = []
    if isinstance(losses, list) and losses:
        steps = list(range(len(losses)))
        (loss_line,) = ax.plot(steps, losses, color=CYAN, linewidth=2.4, label="SFT loss")
        legends.append(loss_line)
        ax.set_ylabel("Cross-Entropy Loss")
    else:
        ax.set_ylabel("Training Metric")
    ax.set(title="LexCrisis Training Curves", xlabel="Training Step")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True)
    if isinstance(rl_rewards, list) and rl_rewards:
        reward_axis = ax.twinx()
        reward_steps = list(range(len(rl_rewards)))
        (reward_line,) = reward_axis.plot(
            reward_steps,
            rl_rewards,
            color=CORAL,
            linewidth=2.0,
            label="GRPO mean reward",
        )
        reward_axis.set_ylabel("Mean Step Reward")
        reward_axis.tick_params(axis="y", colors=TEXT)
        legends.append(reward_line)
    ax.spines[["top", "right"]].set_visible(False)
    if legends:
        ax.legend(legends, [line.get_label() for line in legends], fontsize=10, loc="best")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_loss.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()


def plot_reward_curve(rows: List[Dict[str, Any]]) -> None:
    randomized = [row for row in rows if row["suite"] == "randomized"]
    if not randomized:
        raise ValueError("episode_rows.json must contain randomized-suite rows for reward curves.")
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for row in randomized:
        grouped.setdefault((row["run_name"], row["task_id"]), []).append(row)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = [CYAN, GREEN, AMBER, CORAL]
    for index, ((run_name, task_id), group) in enumerate(sorted(grouped.items())):
        ordered = sorted(group, key=lambda row: row["seed"])
        xs = list(range(1, len(ordered) + 1))
        ys = [row["final_score"] for row in ordered]
        label = RUN_LABELS.get(run_name, run_name.title())
        ax.plot(xs, ys, marker="o", linewidth=2.1, color=colors[index % len(colors)], label=f"{label} / {task_id}")
    ax.set(
        title="LexCrisis Randomized-Suite Score Curve",
        xlabel="Randomized Evaluation Episode",
        ylabel="Final Task Score",
        ylim=(0.0, 1.0),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "reward_curve.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()


def plot_score_comparison(summary_rows: List[Dict[str, Any]]) -> None:
    fixed_rows = [row for row in summary_rows if row["suite"] == "fixed"]
    if not fixed_rows:
        raise ValueError("summary.json must contain fixed-suite aggregates for score comparison.")
    run_names = sorted({row["run_name"] for row in fixed_rows})
    task_ids = sorted({row["task_id"] for row in fixed_rows})
    labels = [task_id.replace("_", " ").title() for task_id in task_ids]
    x_positions = list(range(len(task_ids)))
    bar_width = 0.8 / max(1, len(run_names))

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = [CYAN, GREEN, AMBER, CORAL]
    for run_index, run_name in enumerate(run_names):
        values = []
        for task_id in task_ids:
            match = next(
                (row for row in fixed_rows if row["run_name"] == run_name and row["task_id"] == task_id),
                None,
            )
            values.append(match["avg_final_score"] if match else 0.0)
        offsets = [pos + (run_index - (len(run_names) - 1) / 2) * bar_width for pos in x_positions]
        ax.bar(
            offsets,
            values,
            bar_width,
            label=RUN_LABELS.get(run_name, run_name.title()),
            color=palette[run_index % len(palette)],
            alpha=0.88,
        )
    ax.set(
        title="LexCrisis Fixed-Suite Score Comparison",
        ylabel="Average Final Score",
        ylim=(0.0, 1.0),
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "score_comparison.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    summary_rows = load_json(args.summary)
    episode_rows = load_json(args.rows)
    training_metrics = load_json(args.training)

    plot_training_loss(training_metrics)
    plot_reward_curve(episode_rows)
    plot_score_comparison(summary_rows)
    print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
