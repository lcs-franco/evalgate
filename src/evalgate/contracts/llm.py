import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from evalgate.errors import InvalidOutputError, validation_error_lines


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
    model_config = ConfigDict(frozen=True)

    raw_text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMClient(Protocol):
    async def complete(self, request: LLMRequest) -> LLMResponse: ...


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    last_raw_output: str | None = None


class StructuredLLM:
    """Validates here rather than in each provider, so adapters and fakes fail the same way."""

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
        self.usage.last_raw_output = response.raw_text

        try:
            parsed_json = json.loads(response.raw_text)
        except json.JSONDecodeError as error:
            reason = f"output is not valid JSON: {error}"
            raise InvalidOutputError(reason, response.raw_text) from None

        try:
            return output_type.model_validate(parsed_json)
        except ValidationError as error:
            problems = "; ".join(validation_error_lines(error))
            reason = f"output does not match {output_type.__name__}: {problems}"
            raise InvalidOutputError(reason, response.raw_text) from None
