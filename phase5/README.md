# Phase 5A — MLflow Tracking on Kubernetes

Objective: introduce model lifecycle discipline (tracking, versioning, reproducibility) on top of the hardened Kubernetes baseline.

## What this phase teaches

1) How to run MLflow tracking server in-cluster
2) How backend/artifact storage choices affect reliability
3) How to validate ingress + TLS for an MLOps control-plane service
4) How to define promotion gates before moving to serving/canary phases

## Files in this folder

- `mlflow-stack.yaml` — namespace, PVC, MLflow deployment/service, TLS cert, ingress
- `phase5-mlflow-bootstrap.sh` — one-command deploy + wait + snapshot

## Deploy

```bash
cd /home/ubuntu/weekly-exercise/kubernetes-exercise
bash phase5/phase5-mlflow-bootstrap.sh
```

## Validate

```bash
export KUBECONFIG=/home/ubuntu/.kube/config
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')

# UI check
curl -k -I --resolve mlflow.local:443:${NODE_IP} https://mlflow.local/

# API check
curl -k --resolve mlflow.local:443:${NODE_IP} \
  -X POST https://mlflow.local/api/2.0/mlflow/experiments/search \
  -H 'Content-Type: application/json' \
  -d '{}'
```

Expected:
- HTTP 200 from `/`
- JSON payload from experiments search API

## Learning checkpoint

- [ ] I can explain backend-store URI and artifact-root decisions.
- [ ] I can prove ingress/TLS path to MLflow works.
- [ ] I can run at least one API call against MLflow tracking endpoint.
- [ ] I can list what must change for production (DB, object store, authn/authz).

## Production caveats (important)

This lab uses SQLite + local PVC for simplicity. For production, move to:
- Postgres/MySQL backend store
- S3-compatible artifact store (MinIO/S3/GCS/Azure Blob)
- External authn/authz and network policies
- Backup/retention and upgrade strategy
