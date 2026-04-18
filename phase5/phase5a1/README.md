# Phase 5A.1 Lab — Baseline vs Fine-tuned Comparison with MLflow (SmolLM2-360M)

Goal: run a complete mini-lifecycle loop:
1) baseline evaluation
2) lightweight fine-tune (LoRA)
3) re-evaluation on same test set
4) comparison and tracking in MLflow

Model used:
- HuggingFaceTB/SmolLM2-360M-Instruct

## Why this lab matters

This is the smallest practical loop that teaches promotion discipline:
- same model family
- same eval set
- two runs (before/after)
- objective metrics recorded

You can now answer: "Did fine-tuning actually improve quality enough to justify promotion?"

## Folder contents

- `data/train.jsonl` — tiny supervised sentiment dataset (40 rows)
- `data/eval.jsonl` — fixed eval dataset (20 rows)
- `run_phase5a1.py` — baseline + LoRA train + eval + MLflow metrics logging
- `run_phase5a1.sh` — orchestration script (port-forward + python run)
- `artifacts/` — local outputs (predictions, comparison, LoRA adapter)

## Execution steps (what happens)

### Step 1: Start MLflow connectivity
`run_phase5a1.sh` port-forwards:
- `svc/mlflow-server` (cluster) -> `127.0.0.1:5001` (local)

Why: keeps the script environment simple and avoids DNS/TLS complexity during lab execution.

### Step 2: Baseline eval run
- Loads base model
- Runs deterministic generation on `data/eval.jsonl`
- Parses output to label (`positive`/`negative`)
- Logs baseline metrics to MLflow experiment `phase5a1-smollm2-360m-sentiment`

### Step 3: LoRA fine-tuning
- Applies LoRA adapters to attention projection layers
- Trains for small fixed steps (`max_steps=20`)
- Saves adapter in `artifacts/lora-adapter/`

### Step 4: Post-tune eval + compare
- Evaluates tuned model on same eval set
- Computes delta against baseline
- Writes `artifacts/comparison.json`

## Run it

### Local JSONL mode (existing)

```bash
cd /home/ubuntu/weekly-exercise/kubernetes-exercise
source .venv/bin/activate
bash phase5/phase5a1/run_phase5a1.sh
```

### Hugging Face dataset mode (new)

```bash
cd /home/ubuntu/weekly-exercise/kubernetes-exercise
source .venv/bin/activate
bash phase5/phase5a1/run_phase5a1.sh \
  --dataset-source hf \
  --hf-dataset rotten_tomatoes \
  --hf-train-split train \
  --hf-eval-split validation \
  --hf-train-limit 80 \
  --hf-eval-limit 40 \
  --max-steps 2
```

Notes:
- `run_phase5a1.sh` now forwards extra CLI flags to `run_phase5a1.py`.
- Use `--hf-train-limit` and `--hf-eval-limit` to keep CPU runs fast.
- For full-size experiments, remove limits (`0` = no cap).

## Expected outputs

Local artifacts:
- `phase5/phase5a1/artifacts/baseline_predictions.csv`
- `phase5/phase5a1/artifacts/finetuned_predictions.csv`
- `phase5/phase5a1/artifacts/comparison.json`
- `phase5/phase5a1/artifacts/lora-adapter/*`

MLflow:
- two runs in experiment `phase5a1-smollm2-360m-sentiment`
  - `baseline`
  - `finetuned_lora`

## Important caveat discovered in this lab

Current Phase 5A MLflow server setup uses local artifact path (`/mlflow-data/artifacts`).
From this client environment, artifact upload API attempts hit permission issues on `/mlflow-data`.

What still works:
- metrics + params logging to MLflow

What is stored locally instead:
- prediction CSVs
- comparison JSON
- LoRA adapter files

Production-quality fix for next iteration:
- configure MLflow artifact store to shared object storage (MinIO/S3)
- ensure artifact upload path is reachable and writable by tracking server

## Learning checkpoint

You should now be able to explain:
- why we compare on same eval set
- why baseline and finetuned runs both matter
- why metrics alone are not enough without artifact/provenance trail
- how CI gates can consume these run metrics for promotion decisions
