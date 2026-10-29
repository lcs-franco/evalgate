import asyncio
import datetime
import secrets
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from evalgate.config import FeatureConfig, load_config, load_feature
from evalgate.contracts import FeatureUnderTest, LLMClient, PromptVersion, StructuredLLM
from evalgate.errors import InvalidOutputError
from evalgate.golden import Case, load_golden_dataset
from evalgate.prompts import load_prompt_version
from evalgate.record import CaseResult, RunRecord, compute_totals, read_git_info
from evalgate.scorers import score


def run_feature(config_path: Path, *, llm: LLMClient, out_dir: Path) -> Path:
    """Raises EvalgateConfigError before any model call when an input file is invalid."""
    feature_dir = config_path.parent

    config = load_config(config_path)
    feature = load_feature(config, feature_dir)

    prompt, prompt_hash = load_prompt_version(
        feature_dir / "prompts",
        config.prompt_version,
        feature.input_type,
        feature.output_type,
    )

    dataset = load_golden_dataset(
        feature_dir / config.golden_dataset, feature.input_type, feature.output_type
    )

    started_at = datetime.datetime.now(datetime.UTC)
    cases = asyncio.run(_run_cases(dataset.cases, feature, prompt, config, llm))

    record = RunRecord(
        run_id=f"{started_at:%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}",
        feature=config.name,
        prompt_version=prompt.id,
        prompt_hash=prompt_hash,
        model=config.model,
        golden_dataset_version=dataset.version,
        git=read_git_info(feature_dir),
        timestamp=started_at,
        cases=cases,
        totals=compute_totals(cases),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    record_path = out_dir / f"{record.run_id}.json"
    record_path.write_text(record.model_dump_json(indent=2))

    return record_path


async def _run_cases(
    cases: list[Case],
    feature: FeatureUnderTest[Any, Any],
    prompt: PromptVersion,
    config: FeatureConfig,
    client: LLMClient,
) -> list[CaseResult]:
    return [await _run_case(case, feature, prompt, config, client) for case in cases]


async def _run_case(
    case: Case,
    feature: FeatureUnderTest[Any, Any],
    prompt: PromptVersion,
    config: FeatureConfig,
    client: LLMClient,
) -> CaseResult:
    structured_llm = StructuredLLM(client, config.model)
    output: BaseModel | None = None
    error_reason: str | None = None

    try:
        output = await feature.run(case.input, prompt, structured_llm)
    except InvalidOutputError as error:
        error_reason = error.reason

    scores = (
        []
        if output is None
        else [score(scorer, case.expected, output) for scorer in config.scorers]
    )

    usage = structured_llm.usage

    return CaseResult(
        id=case.id,
        category=str(getattr(case.expected, config.category_field)),
        expected_output_hash=case.expected_output_hash,
        output=None if output is None else output.model_dump(mode="json"),
        raw_output=usage.last_raw_output,
        scores=scores,
        passed=error_reason is None and all(result.passed for result in scores if result.blocking),
        error=error_reason,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        latency_ms=usage.latency_ms,
    )
