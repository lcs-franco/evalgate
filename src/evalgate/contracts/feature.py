import datetime
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict

from evalgate.contracts.llm import StructuredLLM


class FewShot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input: dict[str, Any]
    output: dict[str, Any]


class PromptVersion(BaseModel):
    """An immutable revision of a feature's prompt, loaded from `prompts/<id>.yaml`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    created_at: datetime.date
    system_prompt: str
    few_shot: tuple[FewShot, ...] = ()


@dataclass(frozen=True)
class FeatureUnderTest[In: BaseModel, Out: BaseModel]:
    """What a consumer exports so evalgate can evaluate its feature."""

    input_type: type[In]
    output_type: type[Out]
    run: Callable[[In, PromptVersion, StructuredLLM], Awaitable[Out]]
