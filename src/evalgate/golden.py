import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from evalgate.errors import EvalgateConfigError, describe_validation_error
from evalgate.files import file_hash, read_yaml


class _RawCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    input: dict[str, Any]
    expected: dict[str, Any]
    expected_difficulty: Literal["easy", "medium", "hard"]
    why: str


@dataclass(frozen=True)
class Case:
    id: str
    input: BaseModel
    expected: BaseModel
    expected_output_hash: str
    expected_difficulty: str
    why: str


@dataclass(frozen=True)
class GoldenDataset:
    version: str
    cases: list[Case]


def expected_output_hash(expected: BaseModel) -> str:
    canonical = json.dumps(expected.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_golden_dataset(
    path: Path, input_type: type[BaseModel], output_type: type[BaseModel]
) -> GoldenDataset:
    """Loads and validates every case, reporting every problem at once."""
    raw = read_yaml(path)
    if not isinstance(raw, list) or not raw:
        raise EvalgateConfigError(f"{path}: expected a non-empty list of cases")

    cases: list[Case] = []
    problems: list[str] = []
    first_seen: dict[str, int] = {}
    for number, entry in enumerate(raw, start=1):
        label = f"case {number}"
        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
            label = f"case {number} ({entry['id']!r})"
        case_problems: list[str] = []
        try:
            raw_case = _RawCase.model_validate(entry)
        except ValidationError as error:
            problems += [f"{label}: {line}" for line in describe_validation_error(error)]
            continue

        if raw_case.id in first_seen:
            case_problems.append(f"duplicate id, already used by case {first_seen[raw_case.id]}")
        else:
            first_seen[raw_case.id] = number
        try:
            case_input = input_type.model_validate(raw_case.input)
        except ValidationError as error:
            case_problems += describe_validation_error(error, "input")
        try:
            expected = output_type.model_validate(raw_case.expected)
        except ValidationError as error:
            case_problems += describe_validation_error(error, "expected")

        if case_problems:
            problems += [f"{label}: {line}" for line in case_problems]
            continue
        cases.append(
            Case(
                id=raw_case.id,
                input=case_input,
                expected=expected,
                expected_output_hash=expected_output_hash(expected),
                expected_difficulty=raw_case.expected_difficulty,
                why=raw_case.why,
            )
        )

    if problems:
        raise EvalgateConfigError(
            f"{path}: invalid golden dataset\n" + "\n".join(f"  - {line}" for line in problems)
        )
    version = file_hash(path)[:12]
    return GoldenDataset(version=version, cases=cases)
