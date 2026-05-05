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

EXPERIMENT_NAME="phase5a1-smollm2-360m-sentiment"
HF_REPO="thecoinmaniac/smollm2-360m-lora-sentiment"
DRY_RUN="false"

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

mkdir -p "$PHASE_DIR"

echo "[1/3] Running promotion gate"
bash "$GATE_DIR/run_gate.sh" --experiment "$EXPERIMENT_NAME"

echo "[2/3] Publishing adapter gate-controlled"
PUBLISH_ARGS=(
  --gate-decision "$GATE_DIR/gate_decision.json"
  --adapter-dir "$ROOT_DIR/phase5/phase5a1/artifacts/lora-adapter"
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

echo "[3/3] Publish manifest: $PHASE_DIR/publish_manifest.json"
