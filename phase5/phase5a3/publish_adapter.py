#!/usr/bin/env python3
"""Publish a promoted Phase 5A.1 LoRA adapter to Hugging Face Hub.

Fail-closed design:
- publication is refused unless gate_decision.json says decision == PROMOTE
- adapter artifacts can be restored from the promoted MLflow finetuned run
- required adapter files must exist before any upload call is made
- HF_TOKEN must be present for a real upload
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
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


def mlflow_get_json(tracking_uri: str, api_path: str, params: dict[str, str]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    url = f"{tracking_uri.rstrip('/')}{api_path}?{query}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def proxy_artifact_url(tracking_uri: str, root_uri: str, artifact_path: str) -> str:
    if not root_uri.startswith("mlflow-artifacts:/"):
        raise ValueError(f"Unsupported MLflow artifact root URI for proxy download: {root_uri}")
    root_rel = root_uri.removeprefix("mlflow-artifacts:/").strip("/")
    full_path = f"{root_rel}/{artifact_path}".strip("/")
    quoted = urllib.parse.quote(full_path, safe="/")
    return f"{tracking_uri.rstrip('/')}/api/2.0/mlflow-artifacts/artifacts/{quoted}"


def mlflow_download_file(
    tracking_uri: str,
    run_id: str,
    artifact_path: str,
    dest: Path,
    root_uri: str | None = None,
) -> None:
    params = urllib.parse.urlencode({"run_id": run_id, "path": artifact_path})
    url = f"{tracking_uri.rstrip('/')}/api/2.0/mlflow/artifacts/get?{params}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            dest.write_bytes(resp.read())
            return
    except urllib.error.HTTPError as exc:
        if exc.code != 404 or not root_uri:
            raise

    # MLflow proxy artifact mode can list artifacts via the tracking API while
    # serving the bytes from the mlflow-artifacts endpoint. This is the path used
    # by the Kubernetes MLflow lab stack.
    with urllib.request.urlopen(proxy_artifact_url(tracking_uri, root_uri, artifact_path), timeout=120) as resp:
        dest.write_bytes(resp.read())


def download_mlflow_artifact_dir(
    *,
    tracking_uri: str,
    run_id: str,
    artifact_path: str,
    output_dir: Path,
) -> list[str]:
    """Recursively download files from an MLflow artifact path into output_dir.

    The Phase 5A.1 runner logs the adapter with artifact_path="lora_adapter".
    MLflow returns file paths including that prefix, so we strip it before writing
    into the local adapter directory expected by the HF publish step.
    """

    downloaded: list[str] = []
    root_uri: str | None = None

    def walk(path: str) -> None:
        nonlocal root_uri
        listing = mlflow_get_json(
            tracking_uri,
            "/api/2.0/mlflow/artifacts/list",
            {"run_id": run_id, "path": path},
        )
        root_uri = root_uri or listing.get("root_uri")
        for item in listing.get("files", []):
            item_path = item.get("path")
            if not item_path:
                continue
            if item.get("is_dir"):
                walk(item_path)
                continue
            rel = item_path
            prefix = artifact_path.rstrip("/") + "/"
            if rel.startswith(prefix):
                rel = rel[len(prefix):]
            dest = output_dir / rel
            mlflow_download_file(tracking_uri, run_id, item_path, dest, root_uri=root_uri)
            downloaded.append(str(dest))

    walk(artifact_path)
    if not downloaded:
        raise RuntimeError(
            f"No files downloaded from MLflow artifact path {artifact_path!r} for run {run_id}"
        )
    return downloaded


def ensure_adapter_available(
    *,
    adapter_dir: Path,
    gate_decision: dict[str, Any],
    tracking_uri: str | None,
    mlflow_artifact_path: str,
) -> dict[str, Any]:
    if adapter_dir.exists():
        validation = validate_adapter_dir(adapter_dir)
        validation["source"] = "local"
        return validation

    if not tracking_uri:
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_dir}. Provide --tracking-uri to restore it from MLflow."
        )

    context = gate_decision.get("context", {}) if isinstance(gate_decision.get("context"), dict) else {}
    finetuned_run_id = context.get("finetuned_run_id")
    if not finetuned_run_id:
        raise RuntimeError("Gate decision is missing context.finetuned_run_id; cannot restore adapter from MLflow")

    downloaded = download_mlflow_artifact_dir(
        tracking_uri=tracking_uri,
        run_id=finetuned_run_id,
        artifact_path=mlflow_artifact_path,
        output_dir=adapter_dir,
    )
    validation = validate_adapter_dir(adapter_dir)
    validation["source"] = "mlflow"
    validation["downloaded_files"] = downloaded
    validation["mlflow_artifact_path"] = mlflow_artifact_path
    return validation


def build_manifest(
    *,
    gate_decision: dict[str, Any],
    adapter_dir: Path,
    hf_repo: str,
    github_sha: str | None,
    dry_run: bool,
    adapter_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validation = adapter_validation or validate_adapter_dir(adapter_dir)
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
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", ""))
    parser.add_argument("--mlflow-artifact-path", default="lora_adapter")
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
        adapter_validation = ensure_adapter_available(
            adapter_dir=adapter_dir,
            gate_decision=gate_decision,
            tracking_uri=args.tracking_uri or None,
            mlflow_artifact_path=args.mlflow_artifact_path,
        )
        manifest = build_manifest(
            gate_decision=gate_decision,
            adapter_dir=adapter_dir,
            hf_repo=args.hf_repo,
            github_sha=os.getenv("GITHUB_SHA"),
            dry_run=args.dry_run,
            adapter_validation=adapter_validation,
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
