#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from pathlib import Path

mlflow = None


def safe_log_artifact(local_path: Path, artifact_path: str):
    try:
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
        return True
    except Exception as e:
        print(f"[WARN] Could not log artifact {local_path.name} to MLflow: {e}")
        return False


def safe_log_artifacts(local_dir: Path, artifact_path: str):
    try:
        mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path)
        return True
    except Exception as e:
        print(f"[WARN] Could not log artifact directory {local_dir} to MLflow: {e}")
        return False

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
EXPERIMENT_NAME = "phase5a1-smollm2-360m-sentiment"


def artifact_uri_requires_proxy_migration(artifact_uri: str | None) -> bool:
    if not artifact_uri:
        return False
    return artifact_uri.startswith("/")


def ensure_proxy_backed_experiment(mlflow_module, experiment_name: str) -> str:
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=mlflow_module.get_tracking_uri())
    desired_artifact_root = f"mlflow-artifacts:/{experiment_name}"
    existing = client.get_experiment_by_name(experiment_name)

    if existing is None:
        client.create_experiment(experiment_name, artifact_location=desired_artifact_root)
        mlflow_module.set_experiment(experiment_name)
        print(f"[INFO] Created experiment {experiment_name} with artifact root {desired_artifact_root}")
        return experiment_name

    if artifact_uri_requires_proxy_migration(existing.artifact_location):
        migrated_name = f"{experiment_name}-proxy"
        migrated = client.get_experiment_by_name(migrated_name)
        if migrated is None:
            migrated_artifact_root = f"mlflow-artifacts:/{migrated_name}"
            client.create_experiment(migrated_name, artifact_location=migrated_artifact_root)
            print(
                f"[WARN] Existing experiment {experiment_name} uses local artifact location "
                f"{existing.artifact_location}. Created {migrated_name} with {migrated_artifact_root}."
            )
        mlflow_module.set_experiment(migrated_name)
        return migrated_name

    mlflow_module.set_experiment(experiment_name)
    return experiment_name


def read_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def map_label_value(label_value):
    if isinstance(label_value, str):
        norm = label_value.strip().lower()
        if norm in {"positive", "pos", "1"}:
            return "positive"
        if norm in {"negative", "neg", "0"}:
            return "negative"
        return None

    if isinstance(label_value, bool):
        return None

    if isinstance(label_value, (int, float)):
        if int(label_value) == 1:
            return "positive"
        if int(label_value) == 0:
            return "negative"
        return None

    return None


def convert_hf_rows_to_binary_rows(rows):
    out = []
    for row in rows:
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        mapped = map_label_value(row.get("label"))
        if mapped is None:
            continue
        out.append({"text": text.strip(), "label": mapped})
    return out


def load_rows_from_hf_dataset(dataset_name: str, config_name: str | None, train_split: str, eval_split: str, train_limit: int, eval_limit: int):
    from datasets import load_dataset

    kwargs = {}
    if config_name:
        kwargs["name"] = config_name

    ds = load_dataset(dataset_name, **kwargs)
    if train_split not in ds:
        raise ValueError(f"train split '{train_split}' not found. Available: {list(ds.keys())}")
    if eval_split not in ds:
        raise ValueError(f"eval split '{eval_split}' not found. Available: {list(ds.keys())}")

    train_rows = convert_hf_rows_to_binary_rows(ds[train_split])
    eval_rows = convert_hf_rows_to_binary_rows(ds[eval_split])

    if train_limit > 0:
        train_rows = train_rows[:train_limit]
    if eval_limit > 0:
        eval_rows = eval_rows[:eval_limit]

    if not train_rows:
        raise ValueError("No usable binary train rows after mapping labels/text")
    if not eval_rows:
        raise ValueError("No usable binary eval rows after mapping labels/text")

    return train_rows, eval_rows


def prompt_for(text: str) -> str:
    return (
        "Classify the sentiment of the text as exactly one word: positive or negative.\n"
        f"Text: {text}\n"
        "Label:"
    )


def normalize_label(text: str) -> str:
    t = text.strip().lower()
    m = re.search(r"\b(positive|negative)\b", t)
    return m.group(1) if m else "unknown"


def evaluate(model, tokenizer, rows, max_new_tokens=3):
    import pandas as pd
    import torch

    model.eval()
    device = next(model.parameters()).device
    preds = []
    latencies = []

    with torch.no_grad():
        for row in rows:
            p = prompt_for(row["text"])
            inputs = tokenizer(p, return_tensors="pt").to(device)
            start = time.perf_counter()
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
            elapsed = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed)

            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            pred = normalize_label(gen_text)
            preds.append(
                {
                    "text": row["text"],
                    "gold": row["label"],
                    "prediction": pred,
                    "raw_output": gen_text.strip(),
                    "correct": int(pred == row["label"]),
                    "latency_ms": round(elapsed, 2),
                }
            )

    df = pd.DataFrame(preds)
    accuracy = float(df["correct"].mean())
    unknown_rate = float((df["prediction"] == "unknown").mean())
    avg_latency_ms = float(df["latency_ms"].mean())
    return {
        "accuracy": accuracy,
        "unknown_rate": unknown_rate,
        "avg_latency_ms": avg_latency_ms,
        "predictions_df": df,
    }


def make_train_texts(rows):
    texts = []
    for r in rows:
        texts.append(f"{prompt_for(r['text'])} {r['label']}")
    return texts


def main():
    global mlflow

    import mlflow
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-source", choices=["local_jsonl", "hf"], default="local_jsonl")
    parser.add_argument("--hf-dataset", default="rotten_tomatoes")
    parser.add_argument("--hf-config", default="")
    parser.add_argument("--hf-train-split", default="train")
    parser.add_argument("--hf-eval-split", default="validation")
    parser.add_argument("--hf-train-limit", type=int, default=0)
    parser.add_argument("--hf-eval-limit", type=int, default=0)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    lab_root = project_root / "phase5" / "phase5a1"
    data_dir = lab_root / "data"
    out_dir = lab_root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_source == "local_jsonl":
        train_rows = read_jsonl(data_dir / "train.jsonl")
        eval_rows = read_jsonl(data_dir / "eval.jsonl")
    else:
        train_rows, eval_rows = load_rows_from_hf_dataset(
            dataset_name=args.hf_dataset,
            config_name=args.hf_config or None,
            train_split=args.hf_train_split,
            eval_split=args.hf_eval_split,
            train_limit=args.hf_train_limit,
            eval_limit=args.hf_eval_limit,
        )

    mlflow.set_tracking_uri(args.tracking_uri)
    active_experiment_name = ensure_proxy_backed_experiment(mlflow, EXPERIMENT_NAME)

    print(f"[INFO] Tracking URI: {args.tracking_uri}")
    print(f"[INFO] Active experiment: {active_experiment_name}")
    print(f"[INFO] Model: {MODEL_ID}")
    print(f"[INFO] Dataset source: {args.dataset_source}")
    if args.dataset_source == "hf":
        print(
            f"[INFO] HF dataset: {args.hf_dataset} config={args.hf_config or '<default>'} "
            f"train_split={args.hf_train_split} eval_split={args.hf_eval_split} "
            f"train_limit={args.hf_train_limit} eval_limit={args.hf_eval_limit}"
        )
    print(f"[INFO] Train samples: {len(train_rows)}, Eval samples: {len(eval_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)

    # Baseline run
    with mlflow.start_run(run_name="baseline") as run:
        mlflow.log_params(
            {
                "stage": "baseline",
                "model_id": MODEL_ID,
                "dataset_source": args.dataset_source,
                "hf_dataset": args.hf_dataset if args.dataset_source == "hf" else "",
                "hf_config": args.hf_config if args.dataset_source == "hf" else "",
                "hf_train_split": args.hf_train_split if args.dataset_source == "hf" else "",
                "hf_eval_split": args.hf_eval_split if args.dataset_source == "hf" else "",
                "eval_samples": len(eval_rows),
            }
        )
        baseline = evaluate(base_model, tokenizer, eval_rows)
        mlflow.log_metrics(
            {
                "accuracy": baseline["accuracy"],
                "unknown_rate": baseline["unknown_rate"],
                "avg_latency_ms": baseline["avg_latency_ms"],
            }
        )
        baseline_csv = out_dir / "baseline_predictions.csv"
        baseline["predictions_df"].to_csv(baseline_csv, index=False)
        mlflow.log_param("baseline_predictions_local_path", str(baseline_csv))
        safe_log_artifact(baseline_csv, artifact_path="predictions")
        baseline_run_id = run.info.run_id
        print(f"[INFO] Baseline run_id: {baseline_run_id} accuracy={baseline['accuracy']:.3f}")

    # LoRA fine-tune
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    ft_model = get_peft_model(base_model, lora_cfg)

    train_texts = make_train_texts(train_rows)
    train_ds = Dataset.from_dict({"text": train_texts})

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=192)

    tokenized = train_ds.map(tok, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=str(out_dir / "lora-output"),
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=ft_model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    with mlflow.start_run(run_name="finetuned_lora") as run:
        mlflow.log_params(
            {
                "stage": "finetuned_lora",
                "base_model_id": MODEL_ID,
                "dataset_source": args.dataset_source,
                "hf_dataset": args.hf_dataset if args.dataset_source == "hf" else "",
                "hf_config": args.hf_config if args.dataset_source == "hf" else "",
                "hf_train_split": args.hf_train_split if args.dataset_source == "hf" else "",
                "hf_eval_split": args.hf_eval_split if args.dataset_source == "hf" else "",
                "baseline_run_id": baseline_run_id,
                "train_samples": len(train_rows),
                "max_steps": args.max_steps,
                "learning_rate": args.learning_rate,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": "q_proj,k_proj,v_proj,o_proj",
            }
        )

        t0 = time.perf_counter()
        trainer.train()
        train_seconds = time.perf_counter() - t0

        adapter_dir = out_dir / "lora-adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        ft_model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        tuned = evaluate(ft_model, tokenizer, eval_rows)
        delta_acc = tuned["accuracy"] - baseline["accuracy"]

        mlflow.log_metrics(
            {
                "accuracy": tuned["accuracy"],
                "unknown_rate": tuned["unknown_rate"],
                "avg_latency_ms": tuned["avg_latency_ms"],
                "delta_accuracy_vs_baseline": delta_acc,
                "train_time_seconds": train_seconds,
            }
        )

        tuned_csv = out_dir / "finetuned_predictions.csv"
        tuned["predictions_df"].to_csv(tuned_csv, index=False)
        mlflow.log_param("finetuned_predictions_local_path", str(tuned_csv))
        safe_log_artifact(tuned_csv, artifact_path="predictions")
        safe_log_artifacts(adapter_dir, artifact_path="lora_adapter")

        compare = {
            "baseline_accuracy": baseline["accuracy"],
            "finetuned_accuracy": tuned["accuracy"],
            "delta_accuracy": delta_acc,
            "baseline_unknown_rate": baseline["unknown_rate"],
            "finetuned_unknown_rate": tuned["unknown_rate"],
        }
        compare_path = out_dir / "comparison.json"
        compare_path.write_text(json.dumps(compare, indent=2))
        mlflow.log_param("comparison_local_path", str(compare_path))
        safe_log_artifact(compare_path, artifact_path="comparison")

        print(
            "[INFO] Finetuned run_id: "
            f"{run.info.run_id} accuracy={tuned['accuracy']:.3f} "
            f"delta={delta_acc:+.3f} train_s={train_seconds:.1f}"
        )


if __name__ == "__main__":
    main()
