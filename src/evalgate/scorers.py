from typing import Literal

from pydantic import BaseModel, ConfigDict


class ExactMatchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["exact_match"]
    field: str
    blocking: bool = True

    @property
    def name(self) -> str:
        return f"exact_match:{self.field}"


# Becomes a discriminated union on `type` when more scorers exist.
ScorerConfig = ExactMatchConfig


class ScoreResult(BaseModel):
    scorer: str
    blocking: bool
    passed: bool


def score(config: ScorerConfig, expected: BaseModel, output: BaseModel) -> ScoreResult:
    passed = getattr(output, config.field) == getattr(expected, config.field)
    return ScoreResult(scorer=config.name, blocking=config.blocking, passed=passed)
