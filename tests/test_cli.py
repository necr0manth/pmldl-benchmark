import json

from vlm_benchmark.cli import main


def test_validate_and_sequential_run_cli(tmp_path, capsys):
    dataset = "tests/fixtures/smoke.jsonl"
    second_config = tmp_path / "second.json"
    second_config.write_text(
        json.dumps(
            {
                "name": "mock-smoke-two",
                "backend": {
                    "type": "mock",
                    "default_response": '{"tool":"idle","args":{},"confidence":0,"abstain":true}',
                },
            }
        ),
        encoding="utf-8",
    )
    assert main(["validate", dataset]) == 0
    assert json.loads(capsys.readouterr().out)["cases"] == 2
    assert (
        main(
            [
                "run",
                dataset,
                "--config",
                str(second_config),
                "--config",
                "configs/mock-smoke.json",
                "--output",
                str(tmp_path),
                "--case-id",
                "smoke_start",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert len(output["runs"]) == 2
    assert (tmp_path / "mock-smoke" / "manifest.json").is_file()
    assert (tmp_path / "mock-smoke-two" / "manifest.json").is_file()
    assert json.loads((tmp_path / "mock-smoke" / "manifest.json").read_text(encoding="utf-8"))["case_ids"] == ["smoke_start"]
