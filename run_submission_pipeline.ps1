param(
    [Parameter(Mandatory = $true)]
    [string]$HfToken,
    [string]$BaseModel = "Qwen/Qwen2.5-1.5B-Instruct",
    [string]$ImprovedModel = "Qwen/Qwen2.5-72B-Instruct"
)

$ErrorActionPreference = "Stop"

function Run-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )
    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name (exit code $LASTEXITCODE)"
    }
}

if ($HfToken -match "xxx_your_real_token" -or $HfToken -match "^hf_xxx") {
    throw "Refusing to run with placeholder HF token. Pass your real token."
}

$env:HF_TOKEN = $HfToken

Run-Step -Name "Collect base traces ($BaseModel)" -Command {
    python collect_traces.py --run-name base --model-name $BaseModel --task-ids task_1 task_2 task_3 --seeds 11 23 37 41 53 --trace-dir outputs/policies/base --verbose-dir outputs/traces/base --no-scripted-hints
}

Run-Step -Name "Collect improved traces ($ImprovedModel)" -Command {
    python collect_traces.py --run-name sft --model-name $ImprovedModel --task-ids task_1 task_2 task_3 --seeds 11 23 37 41 53 --trace-dir outputs/policies/sft --verbose-dir outputs/traces/sft --no-scripted-hints
}

Run-Step -Name "Aggregate evaluation" -Command {
    python evaluate_runs.py --run oracle=scripted --run base=trace_dir:outputs/policies/base --run sft=trace_dir:outputs/policies/sft
}

Run-Step -Name "Generate plots" -Command {
    python generate_plots.py
}

Run-Step -Name "Build judge report" -Command {
    python build_judge_report.py
}

Run-Step -Name "Run submission audit" -Command {
    python submission_audit.py
}

Write-Host ""
Write-Host "Pipeline completed successfully." -ForegroundColor Green
