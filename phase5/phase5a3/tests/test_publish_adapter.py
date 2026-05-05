import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "publish_adapter.py"


def load_module():
    spec = importlib.util.spec_from_file_location("publish_adapter", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def write_adapter_files(adapter_dir: Path):
    adapter_dir.mkdir(parents=True)
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "README.md",
    ):
        (adapter_dir / name).write_text("{}", encoding="utf-8")


def promoted_gate():
    return {
        "decision": "PROMOTE",
        "reasons": [],
        "checks": {"latency_ratio": {"pass": True}},
        "derived": {"latency_ratio_vs_baseline": 1.2},
        "baseline_metrics": {"accuracy": 0.67},
        "finetuned_metrics": {"accuracy": 0.87},
        "context": {
            "experiment": "phase5a1-smollm2-360m-sentiment-proxy",
            "requested_experiment": "phase5a1-smollm2-360m-sentiment",
            "baseline_run_id": "baseline123",
            "finetuned_run_id": "fine456",
        },
    }


def test_ensure_promoted_rejects_non_promote():
    mod = load_module()
    try:
        mod.ensure_promoted({"decision": "REJECT", "reasons": ["latency_ratio"]})
    except RuntimeError as exc:
        assert "Refusing to publish" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


def test_validate_adapter_dir_requires_core_files(tmp_path):
    mod = load_module()
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    try:
        mod.validate_adapter_dir(adapter_dir)
    except FileNotFoundError as exc:
        assert "adapter_model.safetensors" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")


def test_proxy_artifact_url_uses_mlflow_artifacts_endpoint():
    mod = load_module()
    url = mod.proxy_artifact_url(
        "http://127.0.0.1:5002/",
        "mlflow-artifacts:/experiment/run123/artifacts",
        "lora_adapter/adapter_config.json",
    )
    assert url == (
        "http://127.0.0.1:5002/api/2.0/mlflow-artifacts/artifacts/"
        "experiment/run123/artifacts/lora_adapter/adapter_config.json"
    )


def test_build_manifest_contains_gate_context(tmp_path):
    mod = load_module()
    adapter_dir = tmp_path / "adapter"
    write_adapter_files(adapter_dir)

    manifest = mod.build_manifest(
        gate_decision=promoted_gate(),
        adapter_dir=adapter_dir,
        hf_repo="thecoinmaniac/smollm2-360m-lora-sentiment",
        github_sha="abc123",
        dry_run=True,
    )

    assert manifest["published"] is False
    assert manifest["dry_run"] is True
    assert manifest["hf_repo"] == "thecoinmaniac/smollm2-360m-lora-sentiment"
    assert manifest["github_sha"] == "abc123"
    assert manifest["gate"]["decision"] == "PROMOTE"
    assert manifest["gate"]["finetuned_run_id"] == "fine456"
    assert "adapter_model.safetensors" in manifest["adapter_validation"]["present_files"]


def test_dry_run_main_writes_manifest(tmp_path, monkeypatch):
    mod = load_module()
    adapter_dir = tmp_path / "adapter"
    write_adapter_files(adapter_dir)
    gate_path = tmp_path / "gate.json"
    manifest_path = tmp_path / "manifest.json"
    gate_path.write_text(json.dumps(promoted_gate()), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "publish_adapter.py",
            "--gate-decision",
            str(gate_path),
            "--adapter-dir",
            str(adapter_dir),
            "--hf-repo",
            "thecoinmaniac/smollm2-360m-lora-sentiment",
            "--manifest",
            str(manifest_path),
            "--dry-run",
        ],
    )

    assert mod.main() == 0
    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert written["dry_run"] is True
    assert written["gate"]["decision"] == "PROMOTE"
