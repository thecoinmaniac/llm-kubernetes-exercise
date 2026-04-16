#!/usr/bin/env bash
set -euo pipefail

# Phase 1 bootstrap for local Kubernetes lab (single-node k3s)
# Target OS: Ubuntu/Debian with sudo available.

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required" >&2
  exit 1
fi

if ! command -v sudo >/dev/null 2>&1; then
  echo "sudo is required" >&2
  exit 1
fi

if ! command -v k3s >/dev/null 2>&1; then
  echo "[1/4] Installing k3s..."
  curl -sfL https://get.k3s.io | sudo INSTALL_K3S_EXEC='--write-kubeconfig-mode 644 --disable traefik' sh -
else
  echo "[1/4] k3s already installed, skipping"
fi

echo "[2/4] Configuring kubeconfig for current user..."
sudo mkdir -p "$HOME/.kube"
sudo cp /etc/rancher/k3s/k3s.yaml "$HOME/.kube/config"
sudo chown "$(id -u):$(id -g)" "$HOME/.kube/config"

if ! command -v helm >/dev/null 2>&1; then
  echo "[3/4] Installing helm..."
  curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
else
  echo "[3/4] helm already installed, skipping"
fi

echo "[4/4] Verifying cluster..."
export KUBECONFIG="$HOME/.kube/config"
kubectl cluster-info
kubectl get nodes -o wide
kubectl get pods -n kube-system

echo "Done. Phase 1 base cluster setup is complete."
