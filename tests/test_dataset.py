import pytest

from conftest import make_case, write_cases
from vlm_benchmark.dataset import load_datasets


def test_group_cannot_leak_across_splits(tmp_path):
    path = tmp_path / "cases.jsonl"
    write_cases(path, [make_case("a", "same", "dev"), make_case("b", "same", "holdout")])
    with pytest.raises(ValueError, match="leaks across splits"):
        load_datasets([path])


def test_missing_and_nonfinite_state_are_not_silently_coerced(tmp_path):
    case = make_case()
    case["mission_state"] = {"state": "IDLE", "sensor": float("nan")}
    path = tmp_path / "cases.jsonl"
    write_cases(path, [case])
    with pytest.raises(ValueError, match="non-finite"):
        load_datasets([path])
