# LexCrisis — Demo Video Script (≤ 2 Minutes)

> **Recording target:** 1:45 – 2:00. Do NOT exceed 2:00.
> **Resolution:** 1080p, text zoom ≥ 125%.
> **Voice:** calm, confident, single speaker. No filler words.

---

## Pre-Recording Checklist (do this before hitting record)

- [ ] Browser tab 1: Hugging Face Space → `/health` endpoint shows `{"status":"ok","benchmark":"lexcrisis"}`
- [ ] Browser tab 2: `README.md` → scrolled to **Submission Links** section
- [ ] Browser tab 3: `outputs/evals/summary.json` — lines 322–373 (sft task_2 block) visible
- [ ] File viewer: `assets/score_comparison.png` open full-screen
- [ ] File viewer: `assets/training_loss.png` open
- [ ] File viewer: `assets/reward_curve.png` open
- [ ] Terminal: `python submission_audit.py` run and passing (screenshot ready)
- [ ] Clock widget hidden / removed from taskbar

---

## Narration Script (time-coded)

### 0:00 – 0:18 | Hook + Problem Statement
*(Show: blank editor or README title card)*

> "Hi judges, we are Team LexCrisis.
>
> LLMs can answer legal questions — but can they **act** on them?
> Under a deadline. With incomplete information. Against adversarial traps.
>
> That's the gap LexCrisis is built to measure."

---

### 0:18 – 0:45 | Environment Innovation
*(Show: `openenv.yaml` task list → then `lexcrisis_env/env.py` dispatch table briefly → then the reward table in README)*

> "LexCrisis is an OpenEnv benchmark for pharmaceutical product-liability litigation.
>
> It has **three tasks**:
> — Conflict-safe client intake,
> — Privilege review under litigation pressure,
> — and Litigation incident command.
>
> Every step is scored by four independent deterministic verifiers:
> outcome correctness, process compliance, deadline handling, and anti-cheat safety.
>
> Agents must **inspect evidence before taking any score-bearing action**.
> If they skip review, the environment penalises them immediately."

---

### 0:45 – 1:05 | Anti-Hacking Design (your strongest differentiator)
*(Show: reward table in README — zoom on the privilege-waiver trap row: `−0.12`)*

> "The reward structure is shaped specifically against hacking.
>
> Loops cost `−0.01` to `−0.05` per step.
> Missing a deadline costs `−0.05` to `−0.06`.
> And there is a **privilege-waiver trap** — `−0.12`, irreversible —
> triggered when an agent produces documents carelessly during discovery.
>
> An agent that reward-hacks will walk straight into it."

---

### 1:05 – 1:30 | Training Pipeline + Evidence of Improvement
*(Show: `assets/training_loss.png` → then `assets/score_comparison.png` → then `summary.json` sft task_2 block)*

> "The pipeline is fully reproducible:
> self-improvement data generation, SFT training with Unsloth and TRL,
> rollout trace collection for base and SFT separately,
> aggregate evaluation, then plot generation.
>
> From our artifacts: base scores 0.001 on task 2.
> After SFT, we see **measurable uplift to 0.061** on the fixed suite.
> Oracle ceiling is 0.93 — there is clear headroom still to close.
>
> All improvement claims come directly from `collect_traces.py` and
> `evaluate_runs.py` artifacts. Nothing is hardcoded."

---

### 1:30 – 1:48 | Reproducibility + Close
*(Show: README submission links section → Hugging Face Space `/health` live → terminal `python submission_audit.py` output)*

> "Everything is linked in the README:
> the Hugging Face Space, our Colab notebook, the GitHub repository, and this video.
>
> Judges can re-run the full pipeline and verify every artifact
> using a single command: `python submission_audit.py`.
>
> LexCrisis — because legal operations is one of the highest-stakes,
> least-solved domains for agentic AI."

---

## Screen Focus Order (cut guide)

| Time      | What's on screen                                         |
|-----------|----------------------------------------------------------|
| 0:00–0:18 | README title / LexCrisis logo / blank editor             |
| 0:18–0:38 | `openenv.yaml` task list (zoom in on task names)         |
| 0:38–0:45 | README reward table — full rows visible                  |
| 0:45–1:05 | README reward table — **privilege-waiver trap row** zoomed |
| 1:05–1:15 | `assets/training_loss.png`                               |
| 1:15–1:25 | `assets/score_comparison.png`                            |
| 1:25–1:30 | `summary.json` — sft task_2 block (`avg_final_score: 0.0609`) |
| 1:30–1:40 | README submission links section                          |
| 1:40–1:48 | HF Space `/health` live + `submission_audit.py` output   |

---

## Claim Safety Rules — Do NOT Say These

| ❌ Do NOT say                                          | ✅ Say instead                                           |
|-------------------------------------------------------|----------------------------------------------------------|
| "SFT significantly improves all three tasks"          | "We observe measurable uplift on task 2 specifically"    |
| "Our model achieves X% accuracy"                      | "Our SFT model scores 0.061 on task_2 vs base at 0.001" |
| "We achieve near-oracle performance"                  | "Oracle ceiling is 0.93; our SFT shows early movement"   |
| "Training converges cleanly"                          | "Loss curve shows a downward trend over 65 steps"        |
| "The model learns to avoid the privilege trap"        | "Our anti-cheat signal is tracked; trap avoidance is 1.0 in our artifacts" |
| "We ran X GPU hours"                                  | Omit unless you have exact numbers from train logs       |
| Any claim not in `summary.json` or `training_metrics.json` | Only cite what those files contain                |

---

## Backup Lines (if you stumble)

- **On improvement:** *"The artifact data shows task_2 moving from baseline to measurable SFT output, with oracle providing our performance ceiling."*
- **On anti-cheat:** *"The environment explicitly traps careless discovery behaviour with an irreversible penalty — something you don't see in toy benchmarks."*
- **On reproducibility:** *"One command — `submission_audit.py` — checks every artifact, every link, and every claim."*
