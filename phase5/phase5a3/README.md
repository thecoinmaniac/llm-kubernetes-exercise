# Phase 5A.3 Runbook - Gate-Controlled Hugging Face Publish

Goal:
publish the promoted LoRA adapter only after the MLflow-backed promotion gate returns `PROMOTE`.

This phase turns the gate from a report into a release control.

## Files

- `publish_adapter.py`
  - reads `phase5/phase5a2/gate_decision.json`
  - refuses to publish unless `decision == PROMOTE`
  - validates the local LoRA adapter directory
  - uploads the adapter folder to Hugging Face Hub
  - writes `publish_manifest.json`

- `run_publish.sh`
  - runs the Phase 5A.2 promotion gate first
  - calls `publish_adapter.py` only if the gate exits successfully

- `publish_manifest.json`
  - generated audit artifact with metrics, gate checks, run IDs, target repo, and GitHub SHA

## Target repos

Model adapter repo:
- `thecoinmaniac/smollm2-360m-lora-sentiment`

Dataset repo used by training lab:
- `thecoinmaniac/llm-k8s-sentiment-datasets`

## Local dry-run validation

Run from repository root:

```bash
source .venv/bin/activate
pip install huggingface_hub pyyaml
bash phase5/phase5a3/run_publish.sh --dry-run
```

What this proves:
- MLflow is reachable
- latest baseline and finetuned runs pass the gate
- adapter files are present
- manifest generation works
- no upload is attempted

## Real publish

Requires `HF_TOKEN` in the environment with write access to the target model repo.

```bash
source .venv/bin/activate
export HF_TOKEN=<token>
bash phase5/phase5a3/run_publish.sh \
  --hf-repo thecoinmaniac/smollm2-360m-lora-sentiment
```

## GitHub Actions

Workflow:
- `.github/workflows/ci-publish.yml`

Manual trigger inputs:
- `experiment_name`
- `hf_model_repo`
- `dry_run`

Recommended order while this lab is still operator-controlled:
1. Run `ci-train`
2. Run `ci-gate`
3. Run `ci-publish`

## Security notes

Keep this workflow manual for now.

The self-hosted runner can access:
- repository files
- GitHub Secrets such as `HF_TOKEN`
- the Kubernetes cluster through kubeconfig
- MLflow through port-forward

Do not automatically publish on every push until runner isolation and release approvals are intentionally designed.

## Failure modes

1. Gate rejects
- `run_publish.sh` stops before calling the publish script.

2. Fresh CI checkout has no local adapter directory
- `phase5/phase5a1/artifacts/` is gitignored and is not present after `actions/checkout`.
- `run_publish.sh` now keeps the MLflow port-forward open after the gate and passes the tracking URI to `publish_adapter.py`.
- `publish_adapter.py` restores the adapter from the promoted finetuned MLflow run artifact path `lora_adapter` before publishing.

3. Missing adapter files after MLflow restore
- `publish_adapter.py` exits non-zero before upload.
- Check that the Phase 5A.1 finetuned run logged `safe_log_artifacts(adapter_dir, artifact_path="lora_adapter")` successfully.

4. Missing `HF_TOKEN`
- real publish exits non-zero; use `--dry-run` for validation-only runs.

5. Wrong HF permissions
- Hugging Face upload fails; verify the token has write access to the model repo.

## Learning checkpoint

- [ ] I can explain why publish must fail closed on non-PROMOTE decisions
- [ ] I can identify which MLflow run IDs produced the published adapter
- [ ] I can inspect `publish_manifest.json` as release evidence
- [ ] I understand why manual dispatch is safer than automatic publishing at this stage
