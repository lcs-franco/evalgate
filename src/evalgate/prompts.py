from pathlib import Path

from pydantic import BaseModel, ValidationError

from evalgate.contracts import PromptVersion
from evalgate.errors import EvalgateConfigError, describe_validation_error
from evalgate.files import config_error, file_hash, read_yaml


def load_prompt_version(
    prompts_dir: Path, version_id: str, input_type: type[BaseModel], output_type: type[BaseModel]
) -> tuple[PromptVersion, str]:
    """Loads `<prompts_dir>/<id>.yaml` and returns it with the hash of the file's content."""
    path = prompts_dir / f"{version_id}.yaml"
    if not path.is_file():
        raise EvalgateConfigError(f"prompt version {version_id!r} not found: {path}")
    try:
        prompt = PromptVersion.model_validate(read_yaml(path))
    except ValidationError as error:
        raise config_error(path, describe_validation_error(error)) from None
    if prompt.id != version_id:
        raise EvalgateConfigError(f"{path}: id {prompt.id!r} does not match the file name")

    problems = []
    for index, shot in enumerate(prompt.few_shot):
        for part, model in (("input", input_type), ("output", output_type)):
            try:
                model.model_validate(getattr(shot, part))
            except ValidationError as error:
                problems += describe_validation_error(error, f"few_shot[{index}].{part}")
    if problems:
        raise config_error(path, problems)
    return prompt, file_hash(path)
