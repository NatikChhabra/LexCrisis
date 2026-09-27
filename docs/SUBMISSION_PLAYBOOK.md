# LexCrisis Submission Playbook (Rubric-Driven)

This file translates the OpenEnv judging docs into an execution checklist for this repo.

## Scoring Priorities

1. Environment Innovation (40%)
2. Storytelling (30%)
3. Showing Improvement in Rewards (20%)
4. Reward + Training Pipeline (10%)

## Non-Negotiable Requirements

- OpenEnv-compliant environment hosted on Hugging Face Space
- runnable training script (Unsloth/HF TRL) in Colab
- real loss/reward evidence from training or trace evaluation
- README linking all judging materials (Space, Colab, repo, video/blog/slides)
- short presentation artifact (<2 min video or equivalent writeup)

## Highest-Impact Strategy For LexCrisis

Do **not** broaden scope right before submission. Maximize trust and clarity:

- prove authentic `base` -> `sft` behavioral improvement
- show verifier-column movement and process compliance changes
- include before/after qualitative traces for all 3 tasks
- keep anti-hacking safeguards explicit (review prerequisite, traps, penalties)

## Commands: Judge-Facing Artifact Pipeline

> Use real model traces. Do not use synthetic/illustrative artifacts for final claims.

```bash
python collect_traces.py --run-name base --model-name Qwen/Qwen2.5-1.5B-Instruct --task-ids task_1 task_2 task_3 --seeds 11 23 37 41 53 --trace-dir outputs/policies/base --verbose-dir outputs/traces/base --no-scripted-hints
python collect_traces.py --run-name sft --model-path outputs/models/sft --task-ids task_1 task_2 task_3 --seeds 11 23 37 41 53 --trace-dir outputs/policies/sft --verbose-dir outputs/traces/sft --no-scripted-hints
python evaluate_runs.py --run oracle=scripted --run base=trace_dir:outputs/policies/base --run sft=trace_dir:outputs/policies/sft
python generate_plots.py
python submission_audit.py
```

If `submission_audit.py` fails, **do not** submit yet.

## “Showing Improvement” Gate

Submission is blocked unless all are true:

- measurable base-vs-sft delta in `outputs/evals/summary.json`
- `base` and `sft` traces are not scripted-equivalent
- readable plots are regenerated from current artifacts
- README narrative matches artifact numbers and trace examples

## Storytelling Structure (What judges should see in 3-5 min)

1. Problem: why legal-ops planning is hard for current LLM agents
2. Environment: what is observed, what actions exist, what gets rewarded
3. Results: base failure -> sft improvement with quantitative + qualitative evidence
4. Why it matters: broader relevance beyond this benchmark

## Red Flags That Lower Score Fast

- claiming “improvement” while base and sft are identical
- missing or placeholder demo link at submission
- mixing synthetic artifacts with judge-facing performance claims
- relying on one scalar and ignoring trace-level behavior
- no explanation of anti-gaming safeguards

