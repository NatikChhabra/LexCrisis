---
title: LexCrisis
emoji: "⚖️"
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - law
  - litigation
  - legal-ops
---

# LexCrisis

LexCrisis trains agents to act like legal incident commanders, not autocomplete systems.

It is an OpenEnv benchmark for high-stakes legal-operations work in pharmaceutical product-liability litigation. The environment is built for verifiable RL: agents act step by step, hidden evidence must be reviewed before score-bearing decisions count, and outcome plus process are graded separately with deterministic verifiers.

## Fast Judge Read (3-5 Minutes)

### 1) Problem

LLMs are still weak at legal operations workflows that require:

- long-horizon planning under deadlines
- partial observability (review-before-decide)
- safety constraints (privilege waiver traps, ethics escalation)
- process correctness, not only final answers

### 2) Environment

LexCrisis provides three tasks (`task_1`, `task_2`, `task_3`) with deterministic graders and step-level verifier signals. Agents must gather evidence, maintain internal findings, and execute compliant legal actions under strict step budgets.

### 3) Results

- The repository includes reproducible evaluation artifacts and plot generation utilities.
- Submission claims are grounded only in authentic model traces (`collect_traces.py`) and aggregate artifacts (`evaluate_runs.py`).
- Run `python submission_audit.py` before submission; do not claim improvement if it fails.

**Current evidence snapshot (from `outputs/evals/summary.json`):**
- `base` fixed-suite scores: `task_1=0.001`, `task_2=0.001`, `task_3=0.001`
- `sft` fixed-suite scores: `task_1=0.001`, `task_2=0.0609`, `task_3=0.001`
- `oracle` fixed-suite scores: `task_1=0.999`, `task_2=0.9335`, `task_3=0.9825`
- This shows measurable uplift on `task_2` in the current artifact set and clear headroom to oracle.

### 4) Why it matters

This benchmark targets a real, underexplored capability gap for agentic systems: compliance-safe multi-step decision making in adversarial professional workflows.

## Submission Links

- Hugging Face Space: [https://huggingface.co/spaces/Natik22may/LexCrisis](https://huggingface.co/spaces/Natik22may/LexCrisis)
- Colab notebook: [https://colab.research.google.com/github/RadheRadheontop/LexCrisis/blob/main/train_lexcrisis.ipynb](https://colab.research.google.com/github/RadheRadheontop/LexCrisis/blob/main/train_lexcrisis.ipynb)
- Code repository: [https://github.com/RadheRadheontop/LexCrisis](https://github.com/RadheRadheontop/LexCrisis)
- Demo video: **[ADD YOUR FINAL YOUTUBE/LOOM URL HERE before submission]** (< 2 minutes; required for judging — see `VIDEO_SCRIPT.md`)

## Why This Is Interesting

- **Original benchmark**: legal conflicts, privilege review, adversarial discovery, deadlines, and ethics constraints are much richer than toy gridworlds.
- **Long-horizon structure**: the agent must gather evidence, update findings, and submit complete work under step budgets.
- **Anti-hacking design**: review is mandatory before score-bearing actions, loops are penalized, timeout without submission is penalized, and privilege-waiver traps are explicit.
- **Verifiable scoring**: every task has deterministic graders and four independent verifier columns:
  - `outcome_correctness`
  - `process_compliance`
  - `deadline_or_latency`
  - `safety_or_anti_cheat`

## Tasks

### `task_1` - Conflict-Safe Client Intake

- Horizon: `24`
- Actions: `review_client`, `check_conflict`, `cite_rule`, `accept_client`, `decline_client`, `submit_intake`
- Verifier focus: conflict-pair F1, decision accuracy, rule citation accuracy

### `task_2` - Privilege Review Under Litigation Pressure

- Horizon: `34`
- Actions: `review_document`, `classify_privilege`, `identify_waiver`, `identify_exception`, `recommend_action`, `submit_review`
- Verifier focus: classification accuracy, doctrine accuracy, waiver F1, exception accuracy, production recommendation accuracy

### `task_3` - Litigation Incident Command

- Horizon: `24`
- Actions: `review_event`, `issue_litigation_hold`, `file_motion`, `respond_discovery`, `assess_expert`, `flag_adversarial`, `flag_ethical_issue`, `submit_triage`, `noop`
- Verifier focus: deadline accuracy, ethics handling, adversarial detection, discovery safety, expert assessment, ordering

## 2-Minute Demo Video Script (Judges)

See `VIDEO_SCRIPT.md` for the full time-coded recording script. Summary:

1. Problem: why legal-ops RL matters.
2. Environment mechanics — tasks, reward table, anti-hacking design.
3. Before/after evidence from `summary.json` and plots.
4. Reproducibility: one command (`python submission_audit.py`) verifies all artifacts.

All narration claims are grounded in `outputs/evals/summary.json` and `outputs/training_metrics.json`.

## Reward Design

Per-step reward is:

```text
reward = grader_score_delta + milestone_bonus + penalty
```

The environment also logs the four verifier columns above on every step and as episode averages.

Key safeguards:

- Agents do not receive positive progress for acting on hidden information.
- Review prerequisite failures raise `review_prerequisite_missing`.
- `respond_discovery` with careless production behavior can trigger the privilege-waiver trap.
- Timeout without submission adds an explicit penalty.

| Signal | Value | Trigger |
|---|---|---|
| Review milestone | +0.02 | First review of any item |
| Correct scoring action | +0.02 to +0.18 | Correct conflict check, classification, etc. |
| Deadline compliance bonus | +0.03 to +0.10 | Action taken before deadline step |
| Loop penalty | -0.01 to -0.05 | Repeating recent actions or noop with overdue deadlines |
| Wrong answer penalty | -0.03 | Incorrect classification, rule, or decision |
| Late deadline penalty | -0.05 to -0.06 | Required action taken after deadline step |
| Timeout without submit | -0.03 | Episode ends at max_steps without terminal action |
| **Privilege-waiver trap** | **-0.12** | `respond_discovery(response_type="produce", objections="")` — irreversible |

## Theme Positioning

LexCrisis is framed as:

- **Primary**: Long-Horizon Planning
- **Secondary**: Self-Improvement
- **Narrative accent**: Wild Card originality for legal-ops

This repo intentionally keeps the submission **single-agent**. Multi-agent legal teams are a future extension, not the current submission claim.

## Honest Evaluation Pipeline

The repo separates four policy roles:

- `oracle`: oracle-reference trajectory
- `base`: unfine-tuned model
- `sft`: supervised fine-tuned model
- `rl`: optional short GRPO refinement

Judge-facing evaluation does **not** inject oracle actions into prompts. The current observation, revealed evidence, findings, and deadlines are the only information shown to the model.

### Evidence provenance policy

To keep judge-facing claims auditable:

- Performance claims must come from `collect_traces.py` -> `evaluate_runs.py` outputs.
- `generate_evidence.py` is for fast local illustration artifacts and should **not** be used as the primary source for base-vs-sft claims.
- Before submission, run `submission_audit.py`; do not submit if it reports failures.

### Current artifact status warning

If `submission_audit.py` reports:

- no measurable `base` vs `sft` delta
- or scripted-equivalent traces for `base` / `sft`

then the current evidence package is **not** judge-ready for “showing improvement” claims.

### Training environment

For training and local checkpoint evaluation, install the model stack in Colab or on a Hugging Face GPU runner:

```bash
pip install datasets transformers trl peft accelerate
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### 1. Generate training data

```bash
python self_improve.py --phase sft --tasks task_1 task_2 task_3
```

This writes:

- `outputs/self_improve/sft_examples.jsonl`

### 2. Train SFT

```bash
python train_sft.py \
  --model unsloth/Qwen2.5-1.5B-Instruct \
  --data outputs/self_improve/sft_examples.jsonl \
  --output-dir outputs/models/sft \
  --metrics outputs/training_metrics.json
```

This writes:

- `outputs/models/sft/`
- `outputs/training_metrics.json`

### 3. Collect rollout traces

Base model:

```bash
python collect_traces.py \
  --run-name base \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --task-ids task_1 task_2 task_3 \
  --seeds 11 23 37 41 53 \
  --trace-dir outputs/policies/base \
  --verbose-dir outputs/traces/base \
  --no-scripted-hints
```

SFT model:

```bash
python collect_traces.py \
  --run-name sft \
  --model-path outputs/models/sft \
  --task-ids task_1 task_2 task_3 \
  --seeds 11 23 37 41 53 \
  --trace-dir outputs/policies/sft \
  --verbose-dir outputs/traces/sft \
  --no-scripted-hints
```

Collected artifacts include:

- `outputs/policies/<run_name>/task_1__fixed.json`
- `outputs/policies/<run_name>/task_1__randomized__11.json`
- `outputs/traces/<run_name>/task_1__fixed__fixed.jsonl`
- `outputs/traces/<run_name>/task_1__fixed__fixed.md`

The `.json` files contain executed actions only. The `.jsonl` files contain rich step traces. The `.md` files render compact judge-ready tables.

### 4. Aggregate evaluation results

```bash
python evaluate_runs.py \
  --run oracle=scripted \
  --run base=trace_dir:outputs/policies/base \
  --run sft=trace_dir:outputs/policies/sft
```

Optional GRPO run:

```bash
python evaluate_runs.py \
  --run oracle=scripted \
  --run base=trace_dir:outputs/policies/base \
  --run sft=trace_dir:outputs/policies/sft \
  --run rl=trace_dir:outputs/policies/rl
```

This writes:

- `outputs/evals/episode_rows.json`
- `outputs/evals/summary.json`

Headline metrics:

- `avg_final_score`
- `verifier_pass_rate`
- `full_episode_solve_rate`
- `review_before_act_rate`
- `deadline_hit_rate`
- `privilege_trap_avoidance_rate`

### 5. Generate plots

```bash
python generate_plots.py
```

Required inputs:

- `outputs/evals/summary.json`
- `outputs/evals/episode_rows.json`
- `outputs/training_metrics.json`

Generated plots:

- `assets/training_loss.png`
- `assets/reward_curve.png`
- `assets/score_comparison.png`

## Optional Short GRPO Pass

This repo includes an experimental short-step GRPO refinement script:

```bash
python train_grpo.py \
  --model-path outputs/models/sft \
  --task-ids task_1 task_3 \
  --output-dir outputs/models/rl \
  --metrics outputs/training_metrics.json \
  --max-train-steps 200
```

Only include `rl` in the final submission story if it improves at least one of:

- `avg_final_score`
- `verifier_pass_rate`
- `review_before_act_rate`
- `deadline_hit_rate`
- `privilege_trap_avoidance_rate`

## Hero Trace Recipe

The strongest trace set for judges is:

- `task_1` fixed: base misses or delays a conflict/rule citation; trained model reviews and cites correctly
- `task_2` fixed: base misses waiver logic for `DOC-006` or `DOC-007`; trained model identifies it and recommends `produce`
- `task_3` fixed: base walks into the privilege trap or misses a deadline; trained model avoids it
- `task_3` randomized seed `11`: trained model succeeds with shuffled order and shifted deadlines

The markdown traces written by `collect_traces.py` already render:

- `step`
- `revealed item`
- `action`
- `feedback`
- `reward`
- `verifier_signals`
- `final_score`

## Local Run

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

Open `http://127.0.0.1:7860` to use the control-room UI.

The UI surfaces:

- last-step reward reason
- verifier columns
- active episode config
- structured findings
- raw state JSON

## Project Layout

```text
lexcrisis/
├── collect_traces.py
├── evaluate_runs.py
├── generate_plots.py
├── inference.py
├── self_improve.py
├── train_grpo.py
├── train_sft.py
├── train_lexcrisis.ipynb
├── lexcrisis_env/
│   ├── env.py
│   ├── graders.py
│   ├── models.py
│   ├── policy_runtime.py
│   ├── prompting.py
│   └── tasks.py
└── server/
    └── ui.html
```

## Final Notes

- Checked-in plot images are presentation assets only until you regenerate them from local artifacts.
- This README intentionally avoids hardcoded trained-model score claims.
- Submission execution checklist is documented in `SUBMISSION_PLAYBOOK.md`.
- We only claim improvements that pass `submission_audit.py` and are backed by non-scripted trace artifacts.
- The final submission flow is:
  1. generate real model traces
  2. aggregate evaluation artifacts
  3. generate plots from those artifacts
  4. link the same Space, Colab, repo, and media assets in the final form

## Submission Quality Gate

Run this command before final submission:

```bash
python submission_audit.py
```

The audit fails fast on common score-killers:

- missing required links/assets
- missing or weak evidence artifacts
- base and sft having no measurable score delta
- scripted-equivalent trace artifacts masquerading as learned behavior

For a strict stop-on-error run in PowerShell:

```powershell
./run_submission_pipeline.ps1 -HfToken "YOUR_REAL_HF_TOKEN"
```

## Plot Readability (Judging Requirement)

Before submission, verify all plots meet the official guidance:

- x-axis and y-axis are labeled with units where relevant
- key comparison curves are on the same axes (baseline vs trained)
- figures are committed as `.png` artifacts
- README includes each key plot with a one-line caption
