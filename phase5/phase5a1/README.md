# Phase 5A.1 Runbook - Baseline vs LoRA Fine-tune Comparison (SmolLM2 + MLflow)

Goal:
run a complete proof loop and produce objective promotion evidence:
1) baseline evaluation
2) LoRA fine-tuning
3) post-tune evaluation on same eval split
4) comparison + MLflow tracking

Model:
- `HuggingFaceTB/SmolLM2-360M-Instruct`

Folder:
- `/home/luna/weekly-exercise/kubernetes-exercise/phase5/phase5a1`

---

## 1) Files and what they do

- `run_phase5a1.sh`
  - orchestration shell script
  - starts temporary MLflow port-forward on localhost:5001
  - invokes Python runner with provided CLI args
  - performs quick endpoint health check

- `run_phase5a1.py`
  - loads dataset (local JSONL or Hugging Face)
  - baseline eval run (`run_name=baseline`)
  - LoRA fine-tune run (`run_name=finetuned_lora`)
  - logs metrics/params/artifacts to MLflow
  - writes local artifacts for audit

- `data/train.jsonl`, `data/eval.jsonl`
  - small local datasets for fast smoke runs

- `artifacts/`
  - generated predictions, comparison JSON, and LoRA adapter output

---

## 2) Environment setup

Run from repository root:
```bash
cd /home/luna/weekly-exercise/kubernetes-exercise
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install transformers==4.45.2 datasets==2.21.0 mlflow==2.16.2 peft==0.12.0 accelerate==0.34.2 scikit-learn pandas sentencepiece
```

What this accomplishes:
- creates isolated runtime
- installs CPU-compatible PyTorch and training/eval/tracking dependencies

Kubernetes context for script:
```bash
export KUBECONFIG=/home/luna/.kube/config
```

What this accomplishes:
- ensures `run_phase5a1.sh` can establish MLflow service port-forward

---

## 3) Run modes

### 3.1 Local JSONL mode (fast smoke)

Run:
```bash
source .venv/bin/activate
bash phase5/phase5a1/run_phase5a1.sh
```

What this does:
- uses `data/train.jsonl` + `data/eval.jsonl`
- good for syntax/logic sanity checks and quick loop validation

### 3.2 Hugging Face sampled mode (fast-ish)

Run:
```bash
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

What this does:
- validates end-to-end logic on a real external dataset with low runtime

### 3.3 Full dataset comparison run

Run:
```bash
source .venv/bin/activate
bash phase5/phase5a1/run_phase5a1.sh \
  --dataset-source hf \
  --hf-dataset rotten_tomatoes \
  --hf-train-split train \
  --hf-eval-split validation \
  --max-steps 20
```

What this does:
- trains on full train split and evaluates on full validation split
- produces strongest evidence for gate decisions

---

## 4) What each run logs to MLflow

Baseline run (`run_name=baseline`):
- metrics: `accuracy`, `unknown_rate`, `avg_latency_ms`
- params: model id, dataset source/splits, eval sample count

Fine-tuned run (`run_name=finetuned_lora`):
- metrics: `accuracy`, `unknown_rate`, `avg_latency_ms`, `delta_accuracy_vs_baseline`, `train_time_seconds`
- params: train sample count, LoRA config, baseline run id linkage

Important behavior:
- runner checks experiment artifact root
- if legacy local-path experiment is detected, it automatically switches to a proxy-backed `-proxy` experiment

---

## 5) Output artifacts and interpretation

Local artifacts:
- `phase5/phase5a1/artifacts/baseline_predictions.csv`
- `phase5/phase5a1/artifacts/finetuned_predictions.csv`
- `phase5/phase5a1/artifacts/comparison.json`
- `phase5/phase5a1/artifacts/lora-adapter/adapter_model.safetensors`
- `phase5/phase5a1/artifacts/lora-adapter/adapter_config.json`

How to interpret key fields:
- `gold`: ground-truth label from eval dataset
- `prediction`: normalized model label (`positive`, `negative`, or `unknown`)
- `raw_output`: direct generated text before normalization
- `correct`: 1 if `prediction == gold`, else 0

---

## 6) Verification commands

### 6.1 Quick MLflow endpoint check

```bash
curl -s -o /dev/null -w 'MLflow HTTP %{http_code}\n' http://127.0.0.1:5001/
```

### 6.2 View latest runs via API

```bash
curl -s -X POST http://127.0.0.1:5001/api/2.0/mlflow/runs/search \
  -H 'Content-Type: application/json' \
  -d '{"max_results":10,"order_by":["attributes.start_time DESC"]}'
```

### 6.3 Check local comparison artifact

```bash
cat /home/luna/weekly-exercise/kubernetes-exercise/phase5/phase5a1/artifacts/comparison.json
```

---

## 7) Common issues and fixes

1) `address already in use` on port 5001
- cause: stale MLflow port-forward process
- fix:
```bash
ss -ltn '( sport = :5001 )'
pkill -f 'kubectl -n mlflow port-forward svc/mlflow-server 5001:5000' || true
```

2) `kubectl` cannot reach cluster from script
- fix:
```bash
export KUBECONFIG=/home/luna/.kube/config
```

3) Long runs look “silent”
- use MLflow API status checks instead of relying only on local process output

4) Artifact upload issues with old experiment roots
- use proxy-backed experiment (`mlflow-artifacts:/...`)
- runner already auto-migrates when legacy local-path root is detected

---

## 8) Learning checkpoint

- [ ] I can run local, sampled-HF, and full-HF modes
- [ ] I can explain baseline vs finetuned metric deltas
- [ ] I can locate and interpret local comparison artifacts
- [ ] I can troubleshoot port-forward and experiment artifact-root issues
- [ ] I can translate final metrics into PROMOTE/REJECT input for next phase

---

## 9) Next phase

Phase 5A.2:
- implement an automated promotion gate script that reads latest MLflow run pair and emits deterministic decision (`PROMOTE`/`REJECT`) from threshold policy.
