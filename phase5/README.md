# Phase 5A Runbook - MLflow Tracking on Kubernetes

This phase adds model lifecycle tracking on top of the hardened cluster baseline.

Primary outcomes:
- deploy MLflow server inside cluster
- expose it via TLS ingress
- validate tracking API before model experiments
- prepare base for Phase 5A.1 (baseline vs fine-tuned compare)

Project folder:
- `$PROJECT_ROOT/phase5` (after setup below)

---

## 1) Files and their purpose

- `mlflow-stack.yaml`
  - defines namespace, PVC, deployment, service, certificate, and ingress
  - configures MLflow artifact proxy mode (`mlflow-artifacts:/`)

- `phase5-mlflow-bootstrap.sh`
  - one-command deployment + readiness waits + snapshot output
  - includes kubeconfig fallback detection for profile/non-profile shells

- `mlflow-proof-flow-architecture.html`
  - architecture diagram showing experiment evidence flow and gate path

- `phase5a1/`
  - practical model comparison lab (baseline eval -> LoRA fine-tune -> compare)

- `phase5a2/`
  - automated promotion gate (`PROMOTE`/`REJECT`) based on MLflow run metrics

---

## 2) Prerequisites

Required foundation from earlier phases:
- k3s cluster healthy
- ingress-nginx installed
- cert-manager installed with `ClusterIssuer/local-selfsigned`

Set location-agnostic project root first:
```bash
cd /home/ubuntu
# If you already cloned previously, skip the next line
git clone https://github.com/thecoinmaniac/llm-kubernetes-exercise.git
cd llm-kubernetes-exercise
export PROJECT_ROOT="$(pwd)"
```

Quick checks:
```bash
cd "$PROJECT_ROOT"
export KUBECONFIG=/home/ubuntu/.kube/config
kubectl get nodes
kubectl -n ingress-nginx get pods
kubectl -n cert-manager get pods
kubectl get clusterissuer local-selfsigned
```

What these checks accomplish:
- ensure control-plane and ingress path are alive
- ensure cert-manager can issue TLS certs

---

## 3) Deploy MLflow stack

Run:
```bash
cd "$PROJECT_ROOT"
bash phase5/phase5-mlflow-bootstrap.sh
```

What the script does step-by-step:
1) applies `phase5/mlflow-stack.yaml`
2) waits for `deploy/mlflow-server` rollout to finish
3) waits for `certificate/mlflow-local-tls` to become Ready
4) prints resource snapshot
5) prints ingress curl hint

---

## 4) Validate resources and endpoint

### 4.1 Kubernetes resource state

Run:
```bash
kubectl -n mlflow get deploy,po,svc,pvc,ingress,certificate -o wide
```

What this confirms:
- deployment available, pod running
- PVC bound
- ingress object present
- certificate ready

### 4.2 Ingress/TLS path validation

Run:
```bash
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')

curl -k -I --resolve mlflow.local:443:${NODE_IP} https://mlflow.local/

curl -k --resolve mlflow.local:443:${NODE_IP} \
  -X POST https://mlflow.local/api/2.0/mlflow/experiments/search \
  -H 'Content-Type: application/json' \
  -d '{"max_results":10}'
```

What this confirms:
- UI endpoint responds over HTTPS
- tracking API responds with experiment JSON

### 4.3 Reliable fallback (port-forward)

Run:
```bash
kubectl -n mlflow port-forward svc/mlflow-server 5001:5000
```

In another shell:
```bash
curl -I http://127.0.0.1:5001/
curl -X POST http://127.0.0.1:5001/api/2.0/mlflow/experiments/search \
  -H 'Content-Type: application/json' \
  -d '{"max_results":10}'
```

What this confirms:
- service itself is healthy even if ingress/DNS route has constraints

---

## 5) Artifact mode note (important)

This stack uses proxy artifact mode to avoid legacy local-path upload problems.

Configured behavior in `mlflow-stack.yaml`:
- `--serve-artifacts`
- `--artifacts-destination /mlflow-data/artifacts`
- `--default-artifact-root mlflow-artifacts:/`

Why it matters:
- clients upload through MLflow API proxy
- avoids client-side writes to local absolute paths like `/mlflow-data/...`

Operational nuance:
- existing legacy experiments keep original artifact roots
- for stable uploads, use proxy-backed experiments (or auto-migrate in runner)

---

## 6) Common issues and fixes

1) `kubectl` tries localhost:8080
- cause: wrong/empty kubeconfig context
- fix:
```bash
export KUBECONFIG=/home/ubuntu/.kube/config
```

2) Ingress curl times out
- isolate with port-forward path first

3) `max_results` API error on experiments search
- always include payload like:
```json
{"max_results":10}
```

4) Port 5001 tunnel/forward issues
- check listener:
```bash
ss -ltn '( sport = :5001 )'
```
- clear stale forward if needed:
```bash
pkill -f 'kubectl -n mlflow port-forward svc/mlflow-server 5001:5000' || true
```

---

## 7) Learning checkpoint

- [ ] I can deploy MLflow stack from script and explain each wait gate
- [ ] I can validate both ingress and port-forward paths
- [ ] I can call MLflow tracking API directly with expected JSON response
- [ ] I understand proxy artifact mode and legacy experiment caveat

---

## 8) Next step

Proceed to Phase 5A.1:
- baseline vs LoRA fine-tune comparison
- objective metric deltas in MLflow
- promotion gate input for next phase
