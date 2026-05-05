#!/usr/bin/env python3
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path


def load_policy(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "thresholds" not in data:
        raise ValueError("Policy file must contain a 'thresholds' mapping")
    return data


def mlflow_post(tracking_uri: str, api_path: str, payload: dict) -> dict:
    url = tracking_uri.rstrip("/") + api_path
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def as_float(metric_value, default=0.0) -> float:
    try:
        return float(metric_value)
    except Exception:
        return float(default)


def metrics_from_run(run: dict) -> dict:
    out = {}
    metrics = run.get("data", {}).get("metrics", [])
    for item in metrics:
        key = item.get("key")
        value = item.get("value")
        if key is None:
            continue
        out[key] = as_float(value)
    return out


def get_experiment_id_by_name(tracking_uri: str, experiment_name: str) -> str | None:
    url = tracking_uri.rstrip("/") + "/api/2.0/mlflow/experiments/get-by-name"
    params = urllib.parse.urlencode({"experiment_name": experiment_name})
    req = urllib.request.Request(f"{url}?{params}", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            exp = data.get("experiment") or {}
            return exp.get("experiment_id")
    except Exception:
        return None


def find_runs(tracking_uri: str, experiment_name: str, run_name: str, max_results: int = 20) -> list[dict]:
    experiment_id = get_experiment_id_by_name(tracking_uri, experiment_name)
    if not experiment_id:
        return []
    payload = {
        "experiment_ids": [experiment_id],
        "filter": f"attributes.status = 'FINISHED' AND tags.mlflow.runName = '{run_name}'",
        "max_results": max_results,
        "order_by": ["attributes.start_time DESC"],
    }
    response = mlflow_post(tracking_uri, "/api/2.0/mlflow/runs/search", payload)
    return response.get("runs", [])


def resolve_experiment_candidates(experiment_name: str) -> list[str]:
    if experiment_name.endswith("-proxy"):
        return [experiment_name]
    # Prefer proxy experiment first because phase5a1 auto-migrates there for artifact safety.
    return [f"{experiment_name}-proxy", experiment_name]

def evaluate_gate(policy: dict, baseline_metrics: dict, finetuned_metrics: dict) -> dict:
    th = policy["thresholds"]
    baseline_acc = as_float(baseline_metrics.get("accuracy"))
    finetuned_acc = as_float(finetuned_metrics.get("accuracy"))
    baseline_unknown = as_float(baseline_metrics.get("unknown_rate"))
    finetuned_unknown = as_float(finetuned_metrics.get("unknown_rate"))
    baseline_latency = as_float(baseline_metrics.get("avg_latency_ms"), default=1.0)
    finetuned_latency = as_float(finetuned_metrics.get("avg_latency_ms"), default=0.0)

    delta_acc = finetuned_acc - baseline_acc
    unknown_increase = finetuned_unknown - baseline_unknown
    latency_ratio = finetuned_latency / baseline_latency if baseline_latency > 0 else 999.0

    checks = {
        "delta_accuracy": {
            "actual": delta_acc,
            "required_min": as_float(th.get("min_delta_accuracy_vs_baseline", 0.0)),
            "pass": delta_acc >= as_float(th.get("min_delta_accuracy_vs_baseline", 0.0)),
        },
        "unknown_rate_increase": {
            "actual": unknown_increase,
            "required_max": as_float(th.get("max_unknown_rate_increase", 0.0)),
            "pass": unknown_increase <= as_float(th.get("max_unknown_rate_increase", 0.0)),
        },
        "latency_ratio": {
            "actual": latency_ratio,
            "required_max": as_float(th.get("max_latency_ratio_vs_baseline", 1.0)),
            "pass": latency_ratio <= as_float(th.get("max_latency_ratio_vs_baseline", 1.0)),
        },
    }

    decision = "PROMOTE" if all(v["pass"] for v in checks.values()) else "REJECT"
    reasons = [name for name, details in checks.items() if not details["pass"]]

    return {
        "decision": decision,
        "reasons": reasons,
        "checks": checks,
        "baseline_metrics": baseline_metrics,
        "finetuned_metrics": finetuned_metrics,
        "derived": {
            "delta_accuracy_vs_baseline": delta_acc,
            "unknown_rate_increase": unknown_increase,
            "latency_ratio_vs_baseline": latency_ratio,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5A.2 promotion gate from MLflow runs")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    parser.add_argument("--experiment", default="phase5a1-smollm2-360m-sentiment")
    parser.add_argument("--baseline-run-name", default="baseline")
    parser.add_argument("--finetuned-run-name", default="finetuned_lora")
    parser.add_argument("--policy", default=str(Path(__file__).with_name("gate_policy.yaml")))
    parser.add_argument("--output", default=str(Path(__file__).with_name("gate_decision.json")))
    args = parser.parse_args()

    policy = load_policy(Path(args.policy))

    baseline_runs = []
    finetuned_runs = []
    selected_experiment = args.experiment
    for exp_name in resolve_experiment_candidates(args.experiment):
        baseline_runs = find_runs(args.tracking_uri, exp_name, args.baseline_run_name)
        finetuned_runs = find_runs(args.tracking_uri, exp_name, args.finetuned_run_name)
        if baseline_runs and finetuned_runs:
            selected_experiment = exp_name
            break

    if not baseline_runs:
        print(
            f"[ERROR] No baseline runs found in experiments: {resolve_experiment_candidates(args.experiment)}",
            file=sys.stderr,
        )
        return 2
    if not finetuned_runs:
        print(
            f"[ERROR] No finetuned runs found in experiments: {resolve_experiment_candidates(args.experiment)}",
            file=sys.stderr,
        )
        return 2

    baseline_run = baseline_runs[0]
    finetuned_run = finetuned_runs[0]

    baseline_metrics = metrics_from_run(baseline_run)
    finetuned_metrics = metrics_from_run(finetuned_run)

    result = evaluate_gate(policy, baseline_metrics, finetuned_metrics)
    result["context"] = {
        "tracking_uri": args.tracking_uri,
        "experiment": selected_experiment,
        "requested_experiment": args.experiment,
        "baseline_run_id": baseline_run.get("info", {}).get("run_id"),
        "finetuned_run_id": finetuned_run.get("info", {}).get("run_id"),
        "policy_path": args.policy,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps({"decision": result["decision"], "reasons": result["reasons"]}))
    if result["decision"] == "REJECT":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
