#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
GATE_DIR="$ROOT_DIR/phase5/phase5a2"

cd "$ROOT_DIR"

if [[ ! -d .venv ]]; then
  echo "[ERROR] .venv not found. Create it and install dependencies first."
  exit 1
fi

source .venv/bin/activate

if [[ -z "${KUBECONFIG:-}" ]]; then
  if [[ -f "$HOME/.kube/config" ]]; then
    export KUBECONFIG="$HOME/.kube/config"
  elif [[ -f "/home/ubuntu/.kube/config" ]]; then
    export KUBECONFIG="/home/ubuntu/.kube/config"
  elif [[ -f "/home/luna/.kube/config" ]]; then
    export KUBECONFIG="/home/luna/.kube/config"
  else
    echo "[ERROR] No kubeconfig found. Set KUBECONFIG explicitly."
    exit 1
  fi
fi

cleanup() {
  if [[ -n "${PF_PID:-}" ]] && kill -0 "$PF_PID" 2>/dev/null; then
    kill "$PF_PID" || true
    wait "$PF_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[1/3] Starting MLflow port-forward on 5001"
PF_LOG="$GATE_DIR/gate-portforward.log"
: > "$PF_LOG"
kubectl -n mlflow port-forward svc/mlflow-server 5001:5000 >"$PF_LOG" 2>&1 &
PF_PID=$!

for _ in $(seq 1 25); do
  if grep -q "Forwarding from" "$PF_LOG"; then
    break
  fi
  if ! kill -0 "$PF_PID" 2>/dev/null; then
    echo "[ERROR] Port-forward process exited early"
    cat "$PF_LOG" || true
    exit 1
  fi
  sleep 1
done

if ! grep -q "Forwarding from" "$PF_LOG"; then
  echo "[ERROR] Port-forward did not become ready"
  cat "$PF_LOG" || true
  exit 1
fi

echo "[2/3] Evaluating promotion gate"
python "$GATE_DIR/gate_decision.py" \
  --tracking-uri "http://127.0.0.1:5001" \
  --policy "$GATE_DIR/gate_policy.yaml" \
  --output "$GATE_DIR/gate_decision.json" \
  "$@"

echo "[3/3] Gate decision artifact: $GATE_DIR/gate_decision.json"