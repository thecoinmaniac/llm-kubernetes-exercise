# Kubernetes LLMOps Lab Runbook (k3s single-node -> MLflow -> baseline vs LoRA compare)

![Repo](https://img.shields.io/badge/repo-llm--kubernetes--exercise-0ea5e9)
![Kubernetes](https://img.shields.io/badge/kubernetes-k3s-326ce5)
![Scope](https://img.shields.io/badge/scope-phase1--phase5a1-22c55e)
![License](https://img.shields.io/badge/license-MIT-f59e0b)

Project root after clone (example):
- `/home/ubuntu/llm-kubernetes-exercise`
- or wherever you clone the repo; this runbook uses `PROJECT_ROOT=$(pwd)` to stay location-agnostic

License:
- MIT (`LICENSE`)

## What this repo helps you do

This repo is an operator-first learning path for running a local Kubernetes MLOps stack end-to-end:
1) bootstrap a k3s single-node cluster
2) install ingress and observability foundations
3) deploy and harden a demo inference service
4) deploy MLflow tracking in-cluster
5) run baseline vs fine-tuned (LoRA) model comparison and log to MLflow

The goal is not only “it works once”, but “you can rerun, verify, debug, and promote with evidence”.

---

## 1) Repository layout

```text
kubernetes-exercise/
├── README.md
├── phase1-bootstrap.sh
├── phase2/
│   ├── phase2-bootstrap.sh
│   └── kube-prometheus-values.yaml
├── phase3/
│   └── inference-stack.yaml
├── phase4/
│   ├── phase4-bootstrap.sh
│   └── hardening.yaml
├── phase5/
│   ├── README.md
│   ├── phase5-mlflow-bootstrap.sh
│   ├── mlflow-stack.yaml
│   ├── mlflow-proof-flow-architecture.html
│   └── phase5a1/
│       ├── README.md
│       ├── run_phase5a1.sh
│       ├── run_phase5a1.py
│       ├── data/
│       └── artifacts/
└── reports/
```

---

## 2) Prerequisites and one-time setup

### 2.1 Host requirements

Recommended minimums for smooth Phase 5A.1 runs:
- 4 vCPU
- 16GB+ RAM
- 30GB+ free disk
- outbound internet access (chart pulls, Python packages, HF model/dataset)

### 2.2 Tooling checks

Run:
```bash
command -v curl
command -v git
command -v kubectl || true
command -v helm || true
python3 --version
```

What this accomplishes:
- verifies shell prerequisites before scripts run
- confirms Python exists for Phase 5A.1 runner
- shows whether kubectl/helm are already present

### 2.3 Clone repository and enter project root

Run:
```bash
cd /home/ubuntu
git clone https://github.com/thecoinmaniac/llm-kubernetes-exercise.git
cd llm-kubernetes-exercise
```

What this accomplishes:
- ensures you are using the canonical upstream repository
- puts you in the repo root where all relative paths in this README are valid

### 2.4 Set project and kubeconfig context

Run:
```bash
export PROJECT_ROOT="$(pwd)"
cd "$PROJECT_ROOT"
export KUBECONFIG=/home/ubuntu/.kube/config
```

What this accomplishes:
- makes project root location-agnostic (works no matter where repo was cloned)
- normalizes working directory for all commands in this README
- prevents kubectl from accidentally targeting the wrong context

---

## 3) Phase 1 - bootstrap k3s single-node cluster

Run:
```bash
bash phase1-bootstrap.sh
```

What this script accomplishes:
- installs/configures k3s control-plane on single node
- prepares kubeconfig for local operator usage
- validates core system readiness before moving to add-ons

Verify after run:
```bash
kubectl cluster-info
kubectl get nodes -o wide
kubectl -n kube-system get pods
kubectl top nodes
```

What each verification checks:
- `cluster-info`: API server reachability
- `get nodes`: node state is `Ready`
- `kube-system pods`: control-plane dependencies are healthy
- `top nodes`: metrics-server path is working

---

## 4) Phase 2 - ingress + observability foundation

Run:
```bash
bash phase2/phase2-bootstrap.sh
```

What this script accomplishes:
- installs `ingress-nginx` in `ingress-nginx` namespace
- installs `kube-prometheus-stack` in `monitoring` namespace
- waits for chart resources to become ready

Verify after run:
```bash
kubectl -n ingress-nginx get pods,svc
kubectl -n monitoring get pods,svc
kubectl get ingressclass
```

What each verification checks:
- ingress controller pods and service are healthy
- Prometheus/Grafana/operator pods are healthy
- `IngressClass` exists so new Ingress objects can bind correctly

---

## 5) Phase 3 - deploy inference-like workload

Run:
```bash
kubectl apply -f phase3/inference-stack.yaml
kubectl -n ml-demo rollout status deploy/ml-demo-inference --timeout=240s
```

What this accomplishes:
- creates deployment/service/ingress for demo workload (`ml-demo` namespace)
- ensures the deployment reached available state before testing traffic

Functional checks:
```bash
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
curl -sS -H 'Host: ml-demo.local' "http://${NODE_IP}/inference/healthz"
curl -sS -H 'Host: ml-demo.local' -X POST "http://${NODE_IP}/inference/predict" \
  -H 'Content-Type: application/json' \
  -d '{"text":"this is good"}'
```

What this checks:
- ingress routing from edge to service
- API behavior on both health and predict endpoints

Metrics checks:
```bash
kubectl -n monitoring port-forward svc/kube-prometheus-stack-prometheus 9090:9090
curl -sS 'http://127.0.0.1:9090/api/v1/query?query=up%7Bjob%3D%22ml-demo-inference%22%7D'
curl -sS 'http://127.0.0.1:9090/api/v1/query?query=inference_requests_total'
```

What this checks:
- `up{job="ml-demo-inference"}` should be 1
- request counter exists and increases as you call predict API

---

## 6) Phase 4 - hardening baseline (TLS, policy, autoscaling)

Run:
```bash
bash phase4/phase4-bootstrap.sh
```

What this script accomplishes:
- installs cert-manager and issues local TLS certs
- applies HPA + ResourceQuota + LimitRange
- applies NetworkPolicy baseline
- applies Prometheus alert rules

Verify hardening resources:
```bash
kubectl -n ml-demo get ingress,certificate,hpa,resourcequota,limitrange,networkpolicy
kubectl -n monitoring get prometheusrule ml-demo-alerts
```

Traffic behavior checks:
```bash
curl -s -o /dev/null -w '%{http_code}\n' -H 'Host: ml-demo.local' "http://${NODE_IP}/inference/healthz"
curl -k -sS --resolve ml-demo.local:443:${NODE_IP} https://ml-demo.local/inference/healthz
```

Expected behavior:
- HTTP path typically redirects to HTTPS (expected in hardened mode)
- HTTPS health endpoint returns success

---

## 7) Phase 5A - deploy MLflow tracking in cluster

Run:
```bash
bash phase5/phase5-mlflow-bootstrap.sh
```

What this script accomplishes:
- applies MLflow stack manifest (namespace, PVC, deployment, service, certificate, ingress)
- waits for deployment rollout and certificate readiness
- prints a resource snapshot for operator verification

Validate MLflow resources:
```bash
kubectl -n mlflow get deploy,po,svc,pvc,ingress,certificate
```

Validate endpoint (ingress path):
```bash
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
curl -k -I --resolve mlflow.local:443:${NODE_IP} https://mlflow.local/
curl -k --resolve mlflow.local:443:${NODE_IP} \
  -X POST https://mlflow.local/api/2.0/mlflow/experiments/search \
  -H 'Content-Type: application/json' \
  -d '{"max_results":10}'
```

What this checks:
- UI path responds over TLS
- tracking API is functional and returns JSON

Fallback when ingress path is constrained:
```bash
kubectl -n mlflow port-forward svc/mlflow-server 5001:5000
curl -I http://127.0.0.1:5001/
curl -X POST http://127.0.0.1:5001/api/2.0/mlflow/experiments/search \
  -H 'Content-Type: application/json' \
  -d '{"max_results":10}'
```

---

## 8) Phase 5A.1 - baseline vs LoRA fine-tune comparison

First-time Python env setup:
```bash
cd "$PROJECT_ROOT"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install transformers==4.45.2 datasets==2.21.0 mlflow==2.16.2 peft==0.12.0 accelerate==0.34.2 scikit-learn pandas sentencepiece
```

What this accomplishes:
- creates isolated runtime for repeatable experiment runs
- installs CPU-safe torch and required model/eval/tracking libraries

Run local JSONL mini-lab:
```bash
source .venv/bin/activate
bash phase5/phase5a1/run_phase5a1.sh
```

Run Hugging Face dataset mode:
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

Run full-size comparison (no limits):
```bash
source .venv/bin/activate
bash phase5/phase5a1/run_phase5a1.sh \
  --dataset-source hf \
  --hf-dataset rotten_tomatoes \
  --hf-train-split train \
  --hf-eval-split validation \
  --max-steps 20
```

What this runner accomplishes:
- starts temporary MLflow port-forward
- executes baseline eval
- applies LoRA fine-tune and re-evals on same split
- logs params/metrics to MLflow
- writes local artifacts (predictions/comparison/adapter)

Artifacts produced:
- `phase5/phase5a1/artifacts/baseline_predictions.csv`
- `phase5/phase5a1/artifacts/finetuned_predictions.csv`
- `phase5/phase5a1/artifacts/comparison.json`
- `phase5/phase5a1/artifacts/lora-adapter/*`

---

## 9) Troubleshooting playbook

### 9.1 Cluster sanity first
```bash
kubectl get nodes
kubectl get pods -A
```

### 9.2 Workload-specific checks
```bash
kubectl -n ml-demo get deploy,po,svc,ingress
kubectl -n mlflow get deploy,po,svc,ingress,pvc,certificate
```

### 9.3 Port-forward conflicts
If a script says port 5001 already in use:
```bash
ss -ltn '( sport = :5001 )'
pkill -f 'kubectl -n mlflow port-forward svc/mlflow-server 5001:5000' || true
```

### 9.4 MLflow artifact upload pitfalls
Current repo is configured for proxy artifact mode. If you still see artifact errors:
- check experiment artifact root in MLflow (legacy experiments may still point to `/mlflow-data/...`)
- use/create proxy-backed experiment (`mlflow-artifacts:/...`)

---

## 10) Learning checkpoints

Mark complete only if you can do these without guesswork:
- [ ] Bootstrap cluster and explain each readiness check
- [ ] Validate ingress and observability end-to-end
- [ ] Deploy and test inference workload via ingress
- [ ] Explain and validate hardening controls interaction
- [ ] Operate MLflow in-cluster and query tracking API
- [ ] Run baseline vs LoRA compare and interpret promotion trade-offs

---

## 11) Phase 5A.2 automated promotion gate

Run:
```bash
source .venv/bin/activate
bash phase5/phase5a2/run_gate.sh --experiment phase5a1-smollm2-360m-sentiment
```

What this accomplishes:
- reads latest `baseline` and `finetuned_lora` runs from MLflow
- applies policy in `phase5/phase5a2/gate_policy.yaml`
- emits `phase5/phase5a2/gate_decision.json`
- returns exit code `0` for PROMOTE and `1` for REJECT (CI-ready)

---

## 12) GitHub Actions + Hugging Face onboarding (manual actions)

To run the new CI workflows, do these one-time actions.

### 12.1 Hugging Face setup

1) Create or sign in to Hugging Face account.
2) Create a public dataset repo (example):
- `<your-hf-username>/llm-k8s-sentiment-datasets`
3) Create a public model repo (example):
- `<your-hf-username>/smollm2-360m-lora-sentiment`
4) Create an HF access token with write scope.

### 12.2 GitHub repository secrets

In GitHub repo settings -> Secrets and variables -> Actions, add:
- `HF_TOKEN` (from Hugging Face)

### 12.3 Self-hosted runner labels

Current workflows expect these labels:
- `self-hosted`
- `linux`
- `train`

Attach these labels to the runner machine that can:
- access your Kubernetes cluster (`kubectl -n mlflow ...`)
- run Python training scripts
- reach Hugging Face endpoints

### 12.4 Available workflows

- `ci-fast` (PR/push): quick unit tests and policy validation
- `ci-train` (manual dispatch): runs Phase 5A.1 training workflow
- `ci-gate` (manual dispatch): runs Phase 5A.2 promotion gate

Then move to Phase 5B:
- publish promoted adapter/model to Hugging Face model repo
- deploy model-serving stack + chat UI with rollout/rollback criteria.
