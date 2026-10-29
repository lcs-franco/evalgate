import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from evalgate.contracts import FeatureUnderTest
from evalgate.errors import EvalgateConfigError, describe_validation_error
from evalgate.files import config_error, read_yaml
from evalgate.scorers import ScorerConfig


class FeatureConfig(BaseModel):
    """`feature.yaml`: how to find and evaluate one feature under test."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    entrypoint: str = Field(pattern=r"^[\w.]+:\w+$", description="`module:attribute`")
    prompt_version: str
    model: str
    golden_dataset: Path
    category_field: str
    scorers: list[ScorerConfig] = Field(min_length=1)


def load_config(path: Path) -> FeatureConfig:
    try:
        return FeatureConfig.model_validate(read_yaml(path))
    except ValidationError as error:
        raise config_error(path, describe_validation_error(error)) from None


def load_feature(config: FeatureConfig, base_dir: Path) -> FeatureUnderTest[Any, Any]:
    """Imports `module:attribute` from the config's directory, as the consumer lays it out."""
    module_name, attribute = config.entrypoint.split(":")
    module_file = base_dir / f"{module_name.replace('.', '/')}.py"
    if not module_file.is_file():
        raise EvalgateConfigError(f"entrypoint {config.entrypoint!r}: {module_file} not found")
    # A name unique to the file, so two features with a module of the same name never collide.
    unique_name = f"evalgate_feature_{hashlib.sha256(str(module_file).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(unique_name, module_file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))  # so the feature can import its own sibling modules
    sys.modules[unique_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise EvalgateConfigError(
            f"entrypoint {config.entrypoint!r}: importing {module_file} failed: {error!r}"
        ) from error

    feature = getattr(module, attribute, None)
    if not isinstance(feature, FeatureUnderTest):
        raise EvalgateConfigError(
            f"entrypoint {config.entrypoint!r}:"
            f" expected a FeatureUnderTest, got {type(feature).__name__}"
        )
    output_fields = feature.output_type.model_fields
    for field in [config.category_field, *(scorer.field for scorer in config.scorers)]:
        if field not in output_fields:
            raise EvalgateConfigError(
                f"field {field!r} is not in {feature.output_type.__name__}"
                f" (fields: {', '.join(output_fields)})"
            )
    return feature
