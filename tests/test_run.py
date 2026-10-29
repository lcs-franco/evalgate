import json
from pathlib import Path
from typing import Any

import pytest
from conftest import CopyExampleFeature
from fakes import ScriptedLLM, golden_case, model_reply

from evalgate.run import run_feature


def run_and_read_record(config: Path, llm: ScriptedLLM, out_dir: Path) -> dict[str, Any]:
    record_path = run_feature(config, llm=llm, out_dir=out_dir)
    record: dict[str, Any] = json.loads(record_path.read_text())
    return record


def test_run_writes_a_run_record_with_scores_and_pass_rates(
    copy_example_feature: CopyExampleFeature, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    monkeypatch.setenv("GITHUB_SHA", "0123456789abcdef0123456789abcdef01234567")
    monkeypatch.setenv("GITHUB_HEAD_REF", "tweak-prompt")
    config = copy_example_feature(
        [
            golden_case("b-1", "Double charge", "billing"),
            golden_case("b-2", "Refund please", "billing"),
            golden_case("t-1", "Export broken", "technical"),
        ],
        model="model-from-config",
    )
    llm = ScriptedLLM(
        {
            "Double charge": model_reply("billing", "Charged twice."),
            "Refund please": model_reply("general"),
            "Export broken": model_reply("technical"),
        }
    )

    record = run_and_read_record(config, llm, tmp_path / "runs")

    assert record["feature"] == "support-email-classifier"
    assert record["prompt_version"] == "v1"
    assert record["model"] == "model-from-config"
    assert record["git"] == {
        "commit": "0123456789abcdef0123456789abcdef01234567",
        "branch": "tweak-prompt",
    }
    assert record["run_id"]
    assert record["timestamp"]
    assert len(record["golden_dataset_version"]) == 12

    first = record["cases"][0]
    assert first["id"] == "b-1"
    assert first["category"] == "billing"
    assert first["output"] == {"category": "billing", "summary": "Charged twice."}
    assert first["scores"] == [{"scorer": "exact_match:category", "blocking": True, "passed": True}]
    assert first["passed"] is True
    assert first["error"] is None
    assert (first["input_tokens"], first["output_tokens"], first["latency_ms"]) == (120, 30, 42.0)
    assert [c["passed"] for c in record["cases"]] == [True, False, True]

    assert record["totals"]["passed"] == 2
    assert record["totals"]["total"] == 3
    assert record["totals"]["pass_rate"] == pytest.approx(2 / 3)
    assert record["totals"]["per_category"] == {
        "billing": {"passed": 1, "total": 2, "pass_rate": 0.5},
        "technical": {"passed": 1, "total": 1, "pass_rate": 1.0},
    }


def test_output_that_cannot_be_parsed_or_validated_fails_the_case_with_the_reason(
    copy_example_feature: CopyExampleFeature, tmp_path: Path
) -> None:
    config = copy_example_feature(
        [
            golden_case("broken-json", "Broken json", "billing"),
            golden_case("bad-category", "Bad category", "billing"),
            golden_case("missing-field", "Missing field", "billing"),
            golden_case("fine", "Fine", "billing"),
        ]
    )
    llm = ScriptedLLM(
        {
            "Broken json": "Sure! The category is billing.",
            "Bad category": model_reply("refunds"),
            "Missing field": '{"category": "billing"}',
            "Fine": model_reply("billing"),
        }
    )

    record = run_and_read_record(config, llm, tmp_path / "runs")

    broken, bad_category, missing, fine = record["cases"]
    for failed in (broken, bad_category, missing):
        assert failed["passed"] is False
        assert failed["output"] is None
        assert failed["scores"] == []
    assert "not valid JSON" in broken["error"]
    assert broken["raw_output"] == "Sure! The category is billing."
    assert (
        "category: Input should be 'billing', 'technical', 'account' or 'general'"
        in (bad_category["error"])
    )
    assert "summary: Field required" in missing["error"]
    assert fine["passed"] is True
    assert record["totals"]["pass_rate"] == 0.25


def test_each_case_records_a_hash_of_its_expected_output_only(
    copy_example_feature: CopyExampleFeature, tmp_path: Path
) -> None:
    config = copy_example_feature(
        [
            golden_case("a", "Alpha", "billing", summary="Same."),
            golden_case(
                "b",
                "Beta",
                "billing",
                summary="Same.",
                why="other note",
                expected_difficulty="hard",
            ),
            golden_case("c", "Gamma", "billing", summary="Different."),
        ]
    )
    llm = ScriptedLLM(
        {
            "Alpha": model_reply("billing"),
            "Beta": model_reply("billing"),
            "Gamma": model_reply("billing"),
        }
    )

    hashes = [
        c["expected_output_hash"]
        for c in run_and_read_record(config, llm, tmp_path / "runs")["cases"]
    ]

    # sha256 of the canonical JSON {"category":"billing","summary":"Same."}
    assert hashes[0] == "9993845eef887d0ff4adc3a77ff7ee629bf784c019b8bbb96d63e2d8507aef27"
    assert hashes[1] == hashes[0]
    assert hashes[2] != hashes[0]


def test_a_run_on_a_pull_request_records_the_head_commit_not_the_merge_commit(
    copy_example_feature: CopyExampleFeature, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"pull_request": {"head": {"sha": "head" + "0" * 36}}}))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_SHA", "merge" + "0" * 35)
    monkeypatch.setenv("GITHUB_HEAD_REF", "tweak-prompt")
    config = copy_example_feature([golden_case("b-1", "Double charge", "billing")])

    record = run_and_read_record(
        config, ScriptedLLM({"Double charge": model_reply("billing")}), tmp_path / "runs"
    )

    assert record["git"] == {"commit": "head" + "0" * 36, "branch": "tweak-prompt"}
