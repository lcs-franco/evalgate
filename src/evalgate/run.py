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
from evalgate.record import CaseResult, RunRecord, git_info, totals
from evalgate.scorers import score


def run_feature(config_path: Path, *, llm: LLMClient, out_dir: Path) -> Path:
    """Makes a run of a feature under test over its golden dataset and writes the run record.

    Raises EvalgateConfigError, before any model call, when the configuration, the prompt version
    or the golden dataset is invalid.
    """
    base_dir = config_path.parent
    config = load_config(config_path)
    feature = load_feature(config, base_dir)
    prompt, prompt_hash = load_prompt_version(
        base_dir / "prompts",
        config.prompt_version,
        feature.input_type,
        feature.output_type,
    )
    dataset = load_golden_dataset(
        base_dir / config.golden_dataset, feature.input_type, feature.output_type
    )

    started = datetime.datetime.now(datetime.UTC)
    cases = asyncio.run(_run_cases(dataset.cases, feature, prompt, config, llm))
    record = RunRecord(
        run_id=f"{started:%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}",
        feature=config.name,
        prompt_version=prompt.id,
        prompt_hash=prompt_hash,
        model=config.model,
        golden_dataset_version=dataset.version,
        git=git_info(base_dir),
        timestamp=started,
        cases=cases,
        totals=totals(cases),
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
    llm = StructuredLLM(client, config.model)
    output: BaseModel | None = None
    failure: str | None = None
    try:
        output = await feature.run(case.input, prompt, llm)
    except InvalidOutputError as error:
        failure = error.reason

    scores = [] if output is None else [score(s, case.expected, output) for s in config.scorers]
    return CaseResult(
        id=case.id,
        category=str(getattr(case.expected, config.category_field)),
        expected_output_hash=case.expected_output_hash,
        output=None if output is None else output.model_dump(mode="json"),
        raw_output=llm.usage.last_raw_output,
        scores=scores,
        passed=failure is None and all(s.passed for s in scores if s.blocking),
        error=failure,
        input_tokens=llm.usage.input_tokens,
        output_tokens=llm.usage.output_tokens,
        latency_ms=llm.usage.latency_ms,
    )
