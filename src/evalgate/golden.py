import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from evalgate.errors import EvalgateConfigError, validation_error_lines
from evalgate.files import file_content_hash, read_yaml


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
    raw_cases = read_yaml(path)

    if not isinstance(raw_cases, list) or not raw_cases:
        raise EvalgateConfigError(f"{path}: expected a non-empty list of cases")

    cases: list[Case] = []
    problems: list[str] = []
    case_number_by_id: dict[str, int] = {}

    for case_number, raw_entry in enumerate(raw_cases, start=1):
        case_label = f"case {case_number}"

        if isinstance(raw_entry, dict) and isinstance(raw_entry.get("id"), str):
            case_label = f"case {case_number} ({raw_entry['id']!r})"

        try:
            raw_case = _RawCase.model_validate(raw_entry)
        except ValidationError as error:
            problems += [f"{case_label}: {line}" for line in validation_error_lines(error)]
            continue

        case_problems: list[str] = []

        if raw_case.id in case_number_by_id:
            first_number = case_number_by_id[raw_case.id]
            case_problems.append(f"duplicate id, already used by case {first_number}")
        else:
            case_number_by_id[raw_case.id] = case_number

        try:
            case_input = input_type.model_validate(raw_case.input)
        except ValidationError as error:
            case_problems += validation_error_lines(error, "input")

        try:
            expected = output_type.model_validate(raw_case.expected)
        except ValidationError as error:
            case_problems += validation_error_lines(error, "expected")

        if case_problems:
            problems += [f"{case_label}: {problem}" for problem in case_problems]
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
        problem_list = "\n".join(f"  - {problem}" for problem in problems)
        raise EvalgateConfigError(f"{path}: invalid golden dataset\n{problem_list}")

    return GoldenDataset(version=file_content_hash(path)[:12], cases=cases)
