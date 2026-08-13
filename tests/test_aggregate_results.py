import json

import pytest

from tools.aggregate_results import build_ranking, collect_runs, main as aggregate_main


def write_summary(root, run_name, value):
    run_dir = root / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"best_map50_95": value}), encoding="utf-8"
    )


def test_aggregate_groups_by_variant_and_ranks(tmp_path):
    write_summary(tmp_path, "d-inverse_ir-raw_seed42", 0.50)
    write_summary(tmp_path, "d-inverse_ir-raw_seed43", 0.60)
    write_summary(tmp_path, "d-log_ir-raw_seed42", 0.30)

    grouped = collect_runs(str(tmp_path), "best_map50_95")
    assert set(grouped) == {"d-inverse_ir-raw", "d-log_ir-raw"}

    ranking = build_ranking(grouped, min_seeds=2)
    assert ranking[0]["variant"] == "d-inverse_ir-raw"
    assert ranking[0]["n"] == 2
    assert ranking[0]["mean"] == pytest.approx(0.55)
    assert ranking[0]["complete"] is True
    assert ranking[1]["variant"] == "d-log_ir-raw"
    assert ranking[1]["complete"] is False


def test_aggregate_main_writes_output(tmp_path):
    write_summary(tmp_path, "a_seed1", 0.4)
    output_path = tmp_path / "ranking.json"

    ranking = aggregate_main([
        "--root", str(tmp_path),
        "--output", str(output_path),
    ])

    assert len(ranking) == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["ranking"][0]["variant"] == "a"


def test_aggregate_fails_without_runs(tmp_path):
    with pytest.raises(RuntimeError, match="No summary.json"):
        aggregate_main(["--root", str(tmp_path)])
