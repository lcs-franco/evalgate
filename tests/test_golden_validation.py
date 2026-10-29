from pathlib import Path

import pytest
from conftest import CopyExampleFeature
from fakes import ScriptedLLM, golden_case

from evalgate.errors import EvalgateConfigError
from evalgate.run import run_feature


def config_error_message(config: Path, out_dir: Path) -> str:
    with pytest.raises(EvalgateConfigError) as error:
        run_feature(config, llm=ScriptedLLM({}), out_dir=out_dir)
    assert not out_dir.exists(), "no run record is written for an invalid golden dataset"
    return str(error.value)


def test_duplicate_case_ids_are_config_error_message(
    copy_example_feature: CopyExampleFeature, tmp_path: Path
) -> None:
    config = copy_example_feature(
        [
            golden_case("b-1", "One", "billing"),
            golden_case("t-1", "Two", "technical"),
            golden_case("b-1", "Three", "billing"),
        ]
    )

    message = config_error_message(config, tmp_path / "runs")

    assert "golden.yaml: invalid golden dataset" in message
    assert "case 3 ('b-1'): duplicate id, already used by case 1" in message


def test_missing_fields_are_config_error_message(
    copy_example_feature: CopyExampleFeature, tmp_path: Path
) -> None:
    no_why = golden_case("b-1", "One", "billing")
    del no_why["why"]
    no_summary = golden_case("b-2", "Two", "billing")
    del no_summary["expected"]["summary"]

    message = config_error_message(copy_example_feature([no_why, no_summary]), tmp_path / "runs")

    assert "case 1 ('b-1'): why: Field required" in message
    assert "case 2 ('b-2'): expected.summary: Field required" in message


def test_unknown_categories_are_config_error_message(
    copy_example_feature: CopyExampleFeature, tmp_path: Path
) -> None:
    config = copy_example_feature(
        [golden_case("b-1", "One", "billing"), golden_case("r-1", "Two", "refunds")]
    )

    message = config_error_message(config, tmp_path / "runs")

    assert (
        "case 2 ('r-1'): expected.category: Input should be "
        "'billing', 'technical', 'account' or 'general'"
    ) in message
    assert "case 1" not in message
