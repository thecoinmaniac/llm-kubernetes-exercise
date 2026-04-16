# Kubernetes MLOps Lab — Phase 1 to Phase 4 (Operator Learning Edition)

![Repo](https://img.shields.io/badge/repo-llm--kubernetes--exercise-0ea5e9)
![Kubernetes](https://img.shields.io/badge/kubernetes-k3s-326ce5)
![Phase](https://img.shields.io/badge/phases-1--4-22c55e)
![License](https://img.shields.io/badge/license-MIT-f59e0b)

Location: `/home/ubuntu/weekly-exercise/kubernetes-exercise`

License: `MIT` (see `LICENSE`)

This project is a hands-on Kubernetes lab designed to build real operator muscle for AI/LLMOps infrastructure.

Instead of only reading concepts, you run a full platform lifecycle:
- bootstrap cluster
- install ingress and observability
- deploy and expose an inference-like service
- add production hardening controls

By the end, you should be able to operate, validate, and troubleshoot a practical K8s baseline for future model-serving phases.

---

## 1) What we are trying to learn (big picture)

### Core learning outcomes

1. Platform bootstrapping discipline
   - Build a clean single-node control plane with reproducible scripts.
   - Understand what “cluster ready” actually means (not just `kubectl get nodes`).

2. Traffic + observability foundations
   - Route app traffic with ingress.
   - Prove metrics scraping end-to-end using Prometheus ServiceMonitor.

3. Inference workload operations
   - Deploy a small API that behaves like a model endpoint (`/predict`, `/healthz`, `/metrics`).
   - Validate behavior through ingress path, not only internal cluster calls.

4. Baseline hardening controls
   - Apply TLS, HPA, ResourceQuota, LimitRange, NetworkPolicy, and alert rules.
   - Understand how security/performance controls interact and fail.

5. Operator-first debugging mindset
   - Move from “it doesn’t work” to a systematic fault-isolation sequence.
   - Distinguish app issues, networking issues, policy issues, and observability issues.

### Why this matters for AI/LLMOps

Real AI systems fail at the platform edges:
- ingress misrouting
- missing/incorrect metrics
- runaway resource usage
- insufficient isolation
- no actionable alerts

This lab gives you a repeatable baseline before moving to Phase 5 workloads like KServe, MLflow, or canary deployments.

---

## 2) Current lab topology (what exists now)

- Kubernetes distro: `k3s` (single-node)
- Kubeconfig: `/home/ubuntu/.kube/config`
- Ingress: `ingress-nginx` (namespace: `ingress-nginx`)
- Observability: `kube-prometheus-stack` (namespace: `monitoring`)
- Demo workload namespace: `ml-demo`
- Demo host: `ml-demo.local`
- Architecture diagram: `/home/ubuntu/weekly-exercise/kubernetes-exercise/kubernetes-lab-phase1-4-architecture.html`

To open the architecture diagram:

```bash
xdg-open /home/ubuntu/weekly-exercise/kubernetes-exercise/kubernetes-lab-phase1-4-architecture.html
```

Phase completion status in this folder:
- Phase 1: complete
- Phase 2: complete
- Phase 3: complete
- Phase 4: complete

Evidence reports are under `reports/`.

---

## 3) Project structure

```
kubernetes-exercise/
├── README.md
├── kubernetes-lab-phase1-4-architecture.html
├── phase1-bootstrap.sh
├── phase2/
│   ├── phase2-bootstrap.sh
│   └── kube-prometheus-values.yaml
├── phase3/
│   └── inference-stack.yaml
├── phase4/
│   ├── phase4-bootstrap.sh
│   └── hardening.yaml
└── reports/
    ├── phase1-setup-verification.txt
    ├── phase2-setup-verification.txt
    ├── phase3-setup-verification.txt
    └── phase4-setup-verification.txt
```

---

## 4) How to run each phase (with purpose)

Set kubeconfig once per shell:

```bash
export KUBECONFIG=$HOME/.kube/config
cd /home/ubuntu/weekly-exercise/kubernetes-exercise
```

### Phase 1 — Base cluster bootstrap

Run:

```bash
bash phase1-bootstrap.sh
```

What this teaches:
- k3s installation and baseline validation
- kubeconfig ownership and local operator setup
- why we disable built-in Traefik when standardizing on ingress-nginx later

Success checks:

```bash
kubectl cluster-info
kubectl get nodes -o wide
kubectl -n kube-system get pods
```

---

### Phase 2 — Ingress + observability foundation

Run:

```bash
bash phase2/phase2-bootstrap.sh
```

What this teaches:
- ingress controller installation and traffic edge standardization
- monitoring stack installation and namespace-level observability
- practical Helm lifecycle (`repo add`, `upgrade --install`, `--wait`)

Success checks:

```bash
kubectl -n ingress-nginx get pods,svc
kubectl -n monitoring get pods,svc
kubectl get ingressclass
```

---

### Phase 3 — Inference-like workload + metrics

Run:

```bash
kubectl apply -f phase3/inference-stack.yaml
kubectl -n ml-demo rollout status deploy/ml-demo-inference --timeout=240s
```

What this teaches:
- deployment/service/ingress wiring
- health/readiness/liveness in runtime behavior
- Prometheus ServiceMonitor discovery with proper labels

Functional validation (replace `<NODE_IP>`):

```bash
curl -sS -H 'Host: ml-demo.local' http://<NODE_IP>/inference/healthz
curl -sS -H 'Host: ml-demo.local' -X POST http://<NODE_IP>/inference/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"this is good"}'
```

Metrics validation:

```bash
kubectl -n monitoring port-forward svc/kube-prometheus-stack-prometheus 9090:9090
curl -sS 'http://127.0.0.1:9090/api/v1/query?query=up%7Bjob%3D%22ml-demo-inference%22%7D'
curl -sS 'http://127.0.0.1:9090/api/v1/query?query=inference_requests_total'
```

Expected:
- `up{job="ml-demo-inference"}` == 1
- `inference_requests_total` exists and increases as you call `/predict`

---

### Phase 4 — Hardening baseline

Run:

```bash
bash phase4/phase4-bootstrap.sh
```

What this teaches:
- TLS issuance and ingress TLS integration
- autoscaling policy with HPA
- namespace guardrails with quotas/limits
- network isolation with default-deny + explicit allow rules
- alert-as-code with PrometheusRule

Validation checks:

```bash
kubectl -n ml-demo get ingress,certificate,hpa,resourcequota,limitrange,networkpolicy
kubectl -n monitoring get prometheusrule ml-demo-alerts
```

HTTP→HTTPS redirect expected:

```bash
curl -s -o /dev/null -w '%{http_code}\n' -H 'Host: ml-demo.local' http://<NODE_IP>/inference/healthz
```

HTTPS functional test:

```bash
curl -k -sS --resolve ml-demo.local:443:<NODE_IP> https://ml-demo.local/inference/healthz
curl -k -sS --resolve ml-demo.local:443:<NODE_IP> -X POST https://ml-demo.local/inference/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"phase4 good"}'
```

---

## 5) Failure modes and debugging playbook

Use this sequence every time:

1) Cluster health first
```bash
kubectl get nodes
kubectl get pods -A
```

2) Workload state
```bash
kubectl -n ml-demo get deploy,po,svc,ingress
kubectl -n ml-demo describe deploy ml-demo-inference
```

3) Ingress routing and edge behavior
```bash
kubectl -n ingress-nginx get pods,svc
kubectl -n ml-demo describe ingress ml-demo-inference
```

4) Metrics discovery path
```bash
kubectl -n monitoring get servicemonitor
kubectl -n monitoring get pods
```

5) Policy and hardening interactions
```bash
kubectl -n ml-demo get networkpolicy
kubectl -n ml-demo get resourcequota,limitrange,hpa
```

Common pitfalls:
- Metrics missing: wrong ServiceMonitor labels or endpoint port mismatch
- HTTPS failing: certificate not Ready or wrong host/SNI in curl
- Direct pod access blocked: expected due to default-deny ingress policy
- HPA `<unknown>`: metrics-server or stabilization lag

---

## 6) Learning checkpoints (self-assessment rubric)

Mark complete only if you can explain and demonstrate each item:

- [ ] I can bootstrap cluster and explain each validation check.
- [ ] I can install ingress + monitoring and verify both are healthy.
- [ ] I can expose an inference API through ingress and test routing.
- [ ] I can prove Prometheus scrapes app metrics and read query results.
- [ ] I can enforce TLS/HPA/quota/policy/alerts and validate each one.
- [ ] I can debug failures with a repeatable operator sequence.

---

## 7) Suggested practice drills (repeatable labs)

Drill A — Metrics break/fix
- Intentionally change ServiceMonitor label.
- Observe `up{job="ml-demo-inference"}` dropping/missing.
- Restore label and verify recovery.

Drill B — Network policy reasoning
- Temporarily tighten policy to block monitoring scrape.
- Observe scrape failures.
- Re-open only required traffic and verify least-privilege operation.

Drill C — Capacity behavior
- Generate load against `/predict`.
- Observe HPA signals and resource pressure.
- Tune requests/limits to reduce instability.

Drill D — TLS operational confidence
- Rotate/recreate certificate.
- Validate ingress still serves expected host with correct secret.

---

## 8) How to use this folder as a learning workbook

For each run:
1. Execute phase script/manifest.
2. Capture command outputs in `reports/` (or a dated subfolder).
3. Write 3 bullets:
   - what changed
   - what failed (if anything)
   - how you validated success

This turns one-time setup into reusable operator knowledge.

---

## 9) Next phase preview

After this baseline, Phase 5 should focus on one of:
- KServe deployment with canary rollout and promotion gates, or
- MLflow model registry + deployable serving workflow with policy checks.

Recommended Phase 5 learning objective:
“Move from static demo inference to versioned model lifecycle with measurable promotion criteria (SLO + error budget + rollback path).”

---

## 10) Quick command reference

```bash
# Enter lab
cd /home/ubuntu/weekly-exercise/kubernetes-exercise
export KUBECONFIG=$HOME/.kube/config

# Re-run phases
bash phase1-bootstrap.sh
bash phase2/phase2-bootstrap.sh
kubectl apply -f phase3/inference-stack.yaml
bash phase4/phase4-bootstrap.sh

# Snapshot state
kubectl get nodes -o wide
kubectl get ns
kubectl -n ml-demo get all
kubectl -n monitoring get pods
```

If this README and the scripts are kept in sync, this folder acts as both:
- execution runbook
- interview-grade learning artifact
- repeatable baseline for Phase 5+ MLOps labs
