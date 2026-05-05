#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PHASE_DIR="$ROOT_DIR/phase5/phase5a3"
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

choose_local_port() {
  for p in 5001 5002 5003 5004 5005 5006 5007 5008 5009 5010; do
    if ! ss -ltn "( sport = :$p )" | grep -q ":$p"; then
      echo "$p"
      return 0
    fi
  done
  return 1
}

EXPERIMENT_NAME="phase5a1-smollm2-360m-sentiment"
HF_REPO="thecoinmaniac/smollm2-360m-lora-sentiment"
DRY_RUN="false"
ADAPTER_DIR="$ROOT_DIR/phase5/phase5a1/artifacts/lora-adapter"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment)
      EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --hf-repo)
      HF_REPO="$2"
      shift 2
      ;;
    --adapter-dir)
      ADAPTER_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      exit 1
      ;;
  esac
done

mkdir -p "$PHASE_DIR" "$GATE_DIR"

LOCAL_PORT="${MLFLOW_LOCAL_PORT:-}"
if [[ -z "$LOCAL_PORT" ]]; then
  LOCAL_PORT="$(choose_local_port)" || {
    echo "[ERROR] No free local port available in range 5001-5010 for MLflow port-forward"
    exit 1
  }
fi

PF_LOG="$PHASE_DIR/publish-portforward.log"
: > "$PF_LOG"

echo "[1/4] Starting MLflow port-forward"
echo "[INFO] Using local MLflow port: $LOCAL_PORT"
kubectl -n mlflow port-forward svc/mlflow-server "$LOCAL_PORT":5000 >"$PF_LOG" 2>&1 &
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

TRACKING_URI="http://127.0.0.1:${LOCAL_PORT}"

echo "[2/4] Running promotion gate"
python "$GATE_DIR/gate_decision.py" \
  --tracking-uri "$TRACKING_URI" \
  --policy "$GATE_DIR/gate_policy.yaml" \
  --output "$GATE_DIR/gate_decision.json" \
  --experiment "$EXPERIMENT_NAME"

echo "[3/4] Publishing adapter gate-controlled"
PUBLISH_ARGS=(
  --gate-decision "$GATE_DIR/gate_decision.json"
  --adapter-dir "$ADAPTER_DIR"
  --tracking-uri "$TRACKING_URI"
  --hf-repo "$HF_REPO"
  --manifest "$PHASE_DIR/publish_manifest.json"
)

if [[ "$DRY_RUN" == "true" ]]; then
  PUBLISH_ARGS+=(--dry-run)
elif [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[ERROR] HF_TOKEN is required unless --dry-run is used"
  exit 1
fi

python "$PHASE_DIR/publish_adapter.py" "${PUBLISH_ARGS[@]}"

echo "[4/4] Publish manifest: $PHASE_DIR/publish_manifest.json"
