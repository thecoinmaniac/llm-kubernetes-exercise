#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${KUBECONFIG:-}" ]]; then
  if [[ -f "$HOME/.kube/config" ]]; then
    export KUBECONFIG="$HOME/.kube/config"
  elif [[ -f "/home/ubuntu/.kube/config" ]]; then
    export KUBECONFIG="/home/ubuntu/.kube/config"
  else
    echo "ERROR: KUBECONFIG not set and no default kubeconfig found (checked $HOME/.kube/config and /home/ubuntu/.kube/config)." >&2
    exit 1
  fi
fi

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[1/5] Applying MLflow namespace + resources"
kubectl apply -f "$BASE_DIR/mlflow-stack.yaml"

echo "[2/5] Waiting for deployment rollout"
kubectl -n mlflow rollout status deploy/mlflow-server --timeout=300s

echo "[3/5] Waiting for TLS certificate readiness"
kubectl -n mlflow wait --for=condition=Ready certificate/mlflow-local-tls --timeout=240s

echo "[4/5] Resource snapshot"
kubectl -n mlflow get deploy,po,svc,pvc,ingress,certificate -o wide

echo "[5/5] Ingress endpoint hint"
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
echo "Use: curl -k --resolve mlflow.local:443:${NODE_IP} https://mlflow.local/"

echo "Phase 5A bootstrap complete."
