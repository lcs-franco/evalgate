import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

EXAMPLE_FEATURE_DIR = Path(__file__).parent.parent / "examples" / "support_email_classifier"

type CopyExampleFeature = Callable[..., Path]


@pytest.fixture
def copy_example_feature(tmp_path: Path) -> CopyExampleFeature:
    def copy(golden_cases: list[dict[str, Any]], **config_overrides: Any) -> Path:
        feature_dir = tmp_path / "feature"
        shutil.copytree(
            EXAMPLE_FEATURE_DIR, feature_dir, ignore=shutil.ignore_patterns("__pycache__")
        )

        (feature_dir / "golden.yaml").write_text(yaml.safe_dump(golden_cases))

        config_path = feature_dir / "feature.yaml"
        config_values = yaml.safe_load(config_path.read_text()) | config_overrides
        config_path.write_text(yaml.safe_dump(config_values))

        return config_path

    return copy
