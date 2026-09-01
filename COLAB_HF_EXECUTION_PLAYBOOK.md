# LexCrisis Execution Playbook (Colab + HF)

This is the single source of truth for what to run next.

## TL;DR Decision

- If you feel stuck/confused: **use Colab T4 first**.
- Do **not** run full training on local CPU.
- For trace collection, use **`--model-path` only** (local loading), not `--model-name`.
- Use Hugging Face Jobs only after Colab succeeds and you want a clean reproducible rerun.

---

## What changed already (done)

The codebase has been upgraded to unblock execution and improve legal benchmark quality:

1. `train_sft.py`
   - Fixed collator crash from string metadata columns.
   - Set right-padding for tokenizer.
   - Added CPU fail-fast to avoid fake "stuck" runs.
2. `lexcrisis_env/policy_runtime.py`
   - `--model-path` now accepts local folders **or model repo ids**.
3. Legal realism upgrades:
   - Added documents `DOC-009`, `DOC-010` (joint-defense and CAPA nuance).
   - Added regulatory event `EVENT-006`.
   - Increased task difficulty/horizon for deeper workflow coverage.
   - Updated scripted baselines and deadline expectations accordingly.

---

## Colab-first path (recommended)

### 0) Create Colab runtime

- Runtime -> Change runtime type -> GPU -> **T4**.
- Start with a clean runtime.

### 1) Setup cell

```bash
%%bash
set -euo pipefail
cd /content
rm -rf LexCrisis
git clone https://github.com/NatikChhabra/LexCrisis.git
cd LexCrisis

pip install -U pip setuptools wheel
pip install -U \
  "openenv-core>=0.1.13" \
  fastapi uvicorn[standard] pydantic requests matplotlib numpy openai \
  datasets transformers trl peft accelerate bitsandbytes
pip install -U "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

python - << 'PY'
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

### 2) Build SFT data

```bash
%%bash
set -euo pipefail
cd /content/LexCrisis
python self_improve.py --phase sft --tasks task_1 task_2 task_3
```

### 3) Train SFT (T4-safe)

```bash
%%bash
set -euo pipefail
cd /content/LexCrisis
python train_sft.py \
  --model "unsloth/Qwen2.5-0.5B-Instruct" \
  --data "outputs/self_improve/sft_examples.jsonl" \
  --output-dir "outputs/models/sft" \
  --metrics "outputs/training_metrics.json" \
  --max-seq-length 512 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --max-steps 80
```

If OOM:

```bash
python train_sft.py \
  --model "unsloth/Qwen2.5-0.5B-Instruct" \
  --data "outputs/self_improve/sft_examples.jsonl" \
  --output-dir "outputs/models/sft" \
  --metrics "outputs/training_metrics.json" \
  --max-seq-length 384 \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --max-steps 60
```

### 4) Collect traces (no API key mode)

```bash
%%bash
set -euo pipefail
cd /content/LexCrisis

python collect_traces.py \
  --run-name base \
  --model-path "unsloth/Qwen2.5-0.5B-Instruct" \
  --task-ids task_1 task_2 task_3 \
  --seeds 11 23 37 41 53 \
  --trace-dir outputs/policies/base \
  --verbose-dir outputs/traces/base \
  --no-scripted-hints

python collect_traces.py \
  --run-name sft \
  --model-path "outputs/models/sft" \
  --task-ids task_1 task_2 task_3 \
  --seeds 11 23 37 41 53 \
  --trace-dir outputs/policies/sft \
  --verbose-dir outputs/traces/sft \
  --no-scripted-hints
```

### 5) Evaluate + plots + audit

```bash
%%bash
set -euo pipefail
cd /content/LexCrisis
python evaluate_runs.py --run oracle=scripted --run base=trace_dir:outputs/policies/base --run sft=trace_dir:outputs/policies/sft
python generate_plots.py
python submission_audit.py
```

### 6) Zip artifacts

```bash
%%bash
set -euo pipefail
cd /content/LexCrisis
zip -r lexcrisis_artifacts.zip outputs assets README.md openenv.yaml || true
ls -lh lexcrisis_artifacts.zip
```

---

## Hugging Face Jobs (only after Colab success)

Use this only for a reproducible rerun once the Colab flow is stable.

1. Authenticate:

```bash
pip install -U "huggingface_hub[cli]"
hf auth login
```

2. Launch a T4 job with your tested commands (same sequence as above).

---

## Common failure mapping

1. Stuck at `0/x` locally
   - Cause: CPU training.
   - Fix: run on Colab T4.

2. `openenv.core` missing
   - Cause: wrong package.
   - Fix: install `openenv-core`.

3. Trace collection asks for API key / remote model unavailable
   - Cause: `--model-name` path.
   - Fix: use `--model-path`.

4. OOM on T4
   - Fix: lower `max-seq-length`, then `gradient-accumulation`, then `max-steps`.

---

## Submission readiness checklist

- [ ] `outputs/models/sft/` exists
- [ ] `outputs/evals/summary.json` exists
- [ ] `outputs/evals/episode_rows.json` exists
- [ ] `outputs/training_metrics.json` exists
- [ ] `assets/training_loss.png` exists
- [ ] `assets/reward_curve.png` exists
- [ ] `assets/score_comparison.png` exists
- [ ] `python submission_audit.py` has no critical fails

