import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from evalgate.errors import InvalidOutputError, describe_validation_error


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    system: str
    messages: tuple[Message, ...]
    output_type: type[BaseModel]


class LLMResponse(BaseModel):
    """What a provider returns: the raw output text, untouched, and what the call cost."""

    model_config = ConfigDict(frozen=True)

    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMClient(Protocol):
    """The provider interface. Adapters hide how each provider asks for structured output."""

    async def complete(self, request: LLMRequest) -> LLMResponse: ...


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    last_raw_output: str | None = None


class StructuredLLM:
    """What a feature under test talks to.

    It fixes the model from configuration, validates the raw output against the requested type
    and records usage, so every provider and every fake goes through the same validation.
    """

    def __init__(self, client: LLMClient, model: str) -> None:
        self._client = client
        self._model = model
        self.usage = Usage()

    async def generate[T: BaseModel](
        self, *, system: str, messages: Sequence[Message], output_type: type[T]
    ) -> T:
        response = await self._client.complete(
            LLMRequest(
                model=self._model,
                system=system,
                messages=tuple(messages),
                output_type=output_type,
            )
        )

        self.usage.input_tokens += response.input_tokens
        self.usage.output_tokens += response.output_tokens
        self.usage.latency_ms += response.latency_ms
        self.usage.last_raw_output = response.text

        try:
            data = json.loads(response.text)
        except json.JSONDecodeError as error:
            raise InvalidOutputError(f"output is not valid JSON: {error}", response.text) from None

        try:
            return output_type.model_validate(data)
        except ValidationError as error:
            problems = "; ".join(describe_validation_error(error))
            reason = f"output does not match {output_type.__name__}: {problems}"
            raise InvalidOutputError(reason, response.text) from None
