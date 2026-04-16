#!/usr/bin/env bash
set -euo pipefail

export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[1/4] Installing cert-manager (if needed)"
helm repo add jetstack https://charts.jetstack.io || true
helm repo update
kubectl create namespace cert-manager --dry-run=client -o yaml | kubectl apply -f -
helm upgrade --install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --set crds.enabled=true \
  --wait --timeout 10m

echo "[2/4] Applying Phase 4 hardening manifests"
kubectl apply -f "$BASE_DIR/hardening.yaml"

echo "[3/4] Waiting for certificate readiness"
kubectl -n ml-demo wait --for=condition=Ready certificate/ml-demo-local-tls --timeout=180s

echo "[4/4] Quick checks"
kubectl -n ml-demo get ingress,certificate,hpa,resourcequota,limitrange,networkpolicy
kubectl -n monitoring get prometheusrule ml-demo-alerts

echo "Phase 4 hardening apply complete."
