import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

EXAMPLE = Path(__file__).parent.parent / "examples" / "support_email_classifier"

type MakeFeature = Callable[..., Path]


@pytest.fixture
def make_feature(tmp_path: Path) -> MakeFeature:
    """Copies the example feature into a temp dir with the given golden cases and config changes."""

    def make(cases: list[dict[str, Any]], **config: Any) -> Path:
        feature_dir = tmp_path / "feature"
        shutil.copytree(EXAMPLE, feature_dir, ignore=shutil.ignore_patterns("__pycache__"))
        (feature_dir / "golden.yaml").write_text(yaml.safe_dump(cases))
        config_path = feature_dir / "feature.yaml"
        settings = yaml.safe_load(config_path.read_text()) | config
        config_path.write_text(yaml.safe_dump(settings))
        return config_path

    return make
