import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "gate_decision.py"


def load_module():
    spec = importlib.util.spec_from_file_location("gate_decision", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_evaluate_gate_promote_case():
    mod = load_module()
    policy = {
        "thresholds": {
            "min_delta_accuracy_vs_baseline": 0.02,
            "max_unknown_rate_increase": 0.00,
            "max_latency_ratio_vs_baseline": 1.15,
        }
    }
    baseline = {"accuracy": 0.70, "unknown_rate": 0.10, "avg_latency_ms": 100}
    finetuned = {"accuracy": 0.73, "unknown_rate": 0.10, "avg_latency_ms": 109}

    out = mod.evaluate_gate(policy, baseline, finetuned)
    assert out["decision"] == "PROMOTE"
    assert out["reasons"] == []


def test_evaluate_gate_reject_with_reasons():
    mod = load_module()
    policy = {
        "thresholds": {
            "min_delta_accuracy_vs_baseline": 0.02,
            "max_unknown_rate_increase": 0.00,
            "max_latency_ratio_vs_baseline": 1.15,
        }
    }
    baseline = {"accuracy": 0.70, "unknown_rate": 0.10, "avg_latency_ms": 100}
    finetuned = {"accuracy": 0.705, "unknown_rate": 0.13, "avg_latency_ms": 130}

    out = mod.evaluate_gate(policy, baseline, finetuned)
    assert out["decision"] == "REJECT"
    assert "delta_accuracy" in out["reasons"]
    assert "unknown_rate_increase" in out["reasons"]
    assert "latency_ratio" in out["reasons"]
