# Phase 5A.2 Runbook - Automated Promotion Gate (MLflow)

Goal:
convert baseline-vs-finetuned evidence into deterministic deployment decisions.

Decision output:
- `PROMOTE` when all policy checks pass
- `REJECT` when any check fails

## Files

- `gate_policy.yaml`
  - threshold policy for promotion checks
- `gate_decision.py`
  - fetches latest MLflow runs and computes decision
  - exits `0` on PROMOTE, `1` on REJECT
- `run_gate.sh`
  - starts temporary MLflow port-forward and runs gate script
- `gate_decision.json`
  - generated output artifact with checks, reasons, and run IDs

## Local execution

```bash
cd /home/ubuntu/llm-kubernetes-exercise
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip pyyaml
bash phase5/phase5a2/run_gate.sh --experiment phase5a1-smollm2-360m-sentiment
```

Interpret result:
- check `phase5/phase5a2/gate_decision.json`
- CI-friendly behavior:
  - exit code 0 => PROMOTE
  - exit code 1 => REJECT

## Policy defaults

- `min_delta_accuracy_vs_baseline: 0.02`
- `max_unknown_rate_increase: 0.00`
- `max_latency_ratio_vs_baseline: 1.15`

Tune thresholds in `gate_policy.yaml` based on your reliability goals.

## CI usage

Workflow: `.github/workflows/ci-gate.yml`

Manual trigger in GitHub Actions:
- Action: `ci-gate`
- Input: experiment name (optional)

Artifact uploaded:
- `phase5a2-gate-decision-<run_id>` containing `gate_decision.json`
