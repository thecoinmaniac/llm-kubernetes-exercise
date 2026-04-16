#!/usr/bin/env bash
set -euo pipefail

export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"

echo "[1/5] Add/refresh Helm repos"
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx || true
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts || true
helm repo update

echo "[2/5] Create namespaces"
kubectl create namespace ingress-nginx --dry-run=client -o yaml | kubectl apply -f -
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

echo "[3/5] Install ingress-nginx"
helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --set controller.service.type=LoadBalancer \
  --wait --timeout 10m

echo "[4/5] Install kube-prometheus-stack"
helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  -f "$(dirname "$0")/kube-prometheus-values.yaml" \
  --wait --timeout 15m

echo "[5/5] Verify"
kubectl get pods -n ingress-nginx
kubectl get pods -n monitoring
kubectl get svc -n ingress-nginx
kubectl get svc -n monitoring

echo "Phase 2 base install complete."
