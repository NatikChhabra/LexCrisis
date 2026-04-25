$ErrorActionPreference = "Stop"

Write-Host "Installing dependencies..." -ForegroundColor Cyan
pip install datasets transformers trl peft accelerate

Write-Host "1) Build SFT data..." -ForegroundColor Cyan
python self_improve.py --phase sft --tasks task_1 task_2 task_3

Write-Host "2) Train SFT..." -ForegroundColor Cyan
python train_sft.py --model unsloth/Qwen2.5-1.5B-Instruct --data outputs/self_improve/sft_examples.jsonl --output-dir outputs/models/sft --metrics outputs/training_metrics.json

Write-Host "3) Collect base traces (Local inference)..." -ForegroundColor Cyan
python collect_traces.py --run-name base --model-path unsloth/Qwen2.5-1.5B-Instruct --task-ids task_1 task_2 task_3 --seeds 11 23 37 41 53 --trace-dir outputs/policies/base --verbose-dir outputs/traces/base --no-scripted-hints

Write-Host "4) Collect SFT traces (Local inference)..." -ForegroundColor Cyan
python collect_traces.py --run-name sft --model-path outputs/models/sft --task-ids task_1 task_2 task_3 --seeds 11 23 37 41 53 --trace-dir outputs/policies/sft --verbose-dir outputs/traces/sft --no-scripted-hints

Write-Host "5) Evaluate + plot + audit..." -ForegroundColor Cyan
python evaluate_runs.py --run oracle=scripted --run base=trace_dir:outputs/policies/base --run sft=trace_dir:outputs/policies/sft
python generate_plots.py
python build_judge_report.py
python submission_audit.py

Write-Host "Local pipeline complete!" -ForegroundColor Green
