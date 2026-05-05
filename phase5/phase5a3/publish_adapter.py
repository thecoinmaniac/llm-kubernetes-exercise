#!/usr/bin/env python3
"""Publish a promoted Phase 5A.1 LoRA adapter to Hugging Face Hub.

Fail-closed design:
- publication is refused unless gate_decision.json says decision == PROMOTE
- required adapter files must exist before any upload call is made
- HF_TOKEN must be present for a real upload
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
)

RECOMMENDED_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "README.md",
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def ensure_promoted(gate_decision: dict[str, Any]) -> None:
    decision = gate_decision.get("decision")
    if decision != "PROMOTE":
        reasons = gate_decision.get("reasons", [])
        raise RuntimeError(f"Refusing to publish: gate decision is {decision!r}, reasons={reasons}")


def validate_adapter_dir(adapter_dir: Path) -> dict[str, list[str]]:
    if not adapter_dir.exists() or not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    missing_required = [name for name in REQUIRED_ADAPTER_FILES if not (adapter_dir / name).is_file()]
    if missing_required:
        raise FileNotFoundError(
            "Adapter directory is missing required files: " + ", ".join(missing_required)
        )

    present = sorted(p.name for p in adapter_dir.iterdir() if p.is_file())
    missing_recommended = [name for name in RECOMMENDED_TOKENIZER_FILES if not (adapter_dir / name).is_file()]
    return {"present_files": present, "missing_recommended_files": missing_recommended}


def build_manifest(
    *,
    gate_decision: dict[str, Any],
    adapter_dir: Path,
    hf_repo: str,
    github_sha: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    validation = validate_adapter_dir(adapter_dir)
    context = gate_decision.get("context", {}) if isinstance(gate_decision.get("context"), dict) else {}
    return {
        "phase": "5A.3",
        "action": "publish_lora_adapter",
        "published": not dry_run,
        "dry_run": dry_run,
        "hf_repo": hf_repo,
        "adapter_dir": str(adapter_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "github_sha": github_sha,
        "gate": {
            "decision": gate_decision.get("decision"),
            "reasons": gate_decision.get("reasons", []),
            "checks": gate_decision.get("checks", {}),
            "derived": gate_decision.get("derived", {}),
            "baseline_metrics": gate_decision.get("baseline_metrics", {}),
            "finetuned_metrics": gate_decision.get("finetuned_metrics", {}),
            "experiment": context.get("experiment"),
            "requested_experiment": context.get("requested_experiment"),
            "baseline_run_id": context.get("baseline_run_id"),
            "finetuned_run_id": context.get("finetuned_run_id"),
        },
        "adapter_validation": validation,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def upload_adapter(adapter_dir: Path, hf_repo: str, token: str, commit_message: str) -> str:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # pragma: no cover - exercised by dependency absence in real env only
        raise RuntimeError("huggingface_hub is required. Install with: pip install huggingface_hub") from exc

    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=hf_repo,
        repo_type="model",
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{hf_repo}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish promoted Phase 5A.1 LoRA adapter to HF Hub")
    parser.add_argument("--gate-decision", default="phase5/phase5a2/gate_decision.json")
    parser.add_argument("--adapter-dir", default="phase5/phase5a1/artifacts/lora-adapter")
    parser.add_argument("--hf-repo", default="thecoinmaniac/smollm2-360m-lora-sentiment")
    parser.add_argument("--manifest", default="phase5/phase5a3/publish_manifest.json")
    parser.add_argument("--commit-message", default="Publish promoted Phase 5A.1 LoRA adapter")
    parser.add_argument("--dry-run", action="store_true", help="Validate and write manifest without uploading")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gate_path = Path(args.gate_decision)
    adapter_dir = Path(args.adapter_dir)
    manifest_path = Path(args.manifest)

    try:
        gate_decision = load_json(gate_path)
        ensure_promoted(gate_decision)
        manifest = build_manifest(
            gate_decision=gate_decision,
            adapter_dir=adapter_dir,
            hf_repo=args.hf_repo,
            github_sha=os.getenv("GITHUB_SHA"),
            dry_run=args.dry_run,
        )

        if not args.dry_run:
            token = os.getenv("HF_TOKEN")
            if not token:
                raise RuntimeError("HF_TOKEN is required for publishing. Use --dry-run for validation only.")
            url = upload_adapter(adapter_dir, args.hf_repo, token, args.commit_message)
            manifest["published_url"] = url

        write_manifest(manifest_path, manifest)
        print(json.dumps({"published": manifest["published"], "dry_run": args.dry_run, "hf_repo": args.hf_repo, "manifest": str(manifest_path)}))
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
