import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "run_phase5a1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_phase5a1", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_map_label_value_supports_common_binary_formats():
    mod = load_module()

    assert mod.map_label_value(1) == "positive"
    assert mod.map_label_value(0) == "negative"
    assert mod.map_label_value("1") == "positive"
    assert mod.map_label_value("0") == "negative"
    assert mod.map_label_value("positive") == "positive"
    assert mod.map_label_value("negative") == "negative"


def test_map_label_value_returns_none_for_non_binary():
    mod = load_module()

    assert mod.map_label_value(2) is None
    assert mod.map_label_value("neutral") is None


def test_convert_hf_rows_to_binary_rows_filters_non_binary_and_missing_text():
    mod = load_module()

    rows = [
        {"text": "great", "label": 1},
        {"text": "bad", "label": 0},
        {"text": "meh", "label": 2},
        {"label": 1},
    ]

    out = mod.convert_hf_rows_to_binary_rows(rows)

    assert out == [
        {"text": "great", "label": "positive"},
        {"text": "bad", "label": "negative"},
    ]


def test_artifact_uri_requires_proxy_migration_for_absolute_paths():
    mod = load_module()

    assert mod.artifact_uri_requires_proxy_migration("/mlflow-data/artifacts/1") is True
    assert mod.artifact_uri_requires_proxy_migration("mlflow-artifacts:/phase5") is False
    assert mod.artifact_uri_requires_proxy_migration(None) is False
