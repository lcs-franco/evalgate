import hashlib
from pathlib import Path
from typing import Any

import yaml

from evalgate.errors import EvalgateConfigError


def read_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text())
    except OSError as error:
        raise EvalgateConfigError(f"{path}: cannot read: {error.strerror}") from None
    except yaml.YAMLError as error:
        raise EvalgateConfigError(f"{path}: not valid YAML: {error}") from None


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_error(path: Path, problems: list[str]) -> EvalgateConfigError:
    return EvalgateConfigError("\n".join(f"{path}: {problem}" for problem in problems))
