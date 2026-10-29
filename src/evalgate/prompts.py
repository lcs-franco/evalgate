from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, ValidationError

from evalgate.contracts import PromptVersion
from evalgate.errors import EvalgateConfigError, validation_error_lines
from evalgate.files import config_error_for_file, file_content_hash, read_yaml


class LoadedPromptVersion(NamedTuple):
    prompt: PromptVersion
    file_hash: str


def load_prompt_version(
    prompts_dir: Path, version_id: str, input_type: type[BaseModel], output_type: type[BaseModel]
) -> LoadedPromptVersion:
    path = prompts_dir / f"{version_id}.yaml"

    if not path.is_file():
        raise EvalgateConfigError(f"prompt version {version_id!r} not found: {path}")

    try:
        prompt = PromptVersion.model_validate(read_yaml(path))
    except ValidationError as error:
        raise config_error_for_file(path, validation_error_lines(error)) from None

    if prompt.id != version_id:
        raise EvalgateConfigError(f"{path}: id {prompt.id!r} does not match the file name")

    problems = []

    for index, example in enumerate(prompt.few_shot):
        for field_name, expected_type in (("input", input_type), ("output", output_type)):
            try:
                expected_type.model_validate(getattr(example, field_name))
            except ValidationError as error:
                location = f"few_shot[{index}].{field_name}"
                problems += validation_error_lines(error, location)

    if problems:
        raise config_error_for_file(path, problems)

    return LoadedPromptVersion(prompt, file_content_hash(path))
