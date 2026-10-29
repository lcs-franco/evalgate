import time
from typing import Any

from openai import AsyncOpenAI
from openai.types.responses import ResponseInputParam, ResponseTextConfigParam

from evalgate.contracts import LLMRequest, LLMResponse


class OpenAIClient:
    """LLM client over OpenAI's Responses API with strict JSON-schema structured output.

    It returns the output text as is: validating it is evalgate's job, not the provider's.
    """

    def __init__(self) -> None:
        self._client = AsyncOpenAI()

    async def complete(self, request: LLMRequest) -> LLMResponse:
        messages: ResponseInputParam = [
            {"role": message.role, "content": message.content} for message in request.messages
        ]
        text: ResponseTextConfigParam = {
            "format": {
                "type": "json_schema",
                "name": request.output_type.__name__,
                "schema": strict_schema(request.output_type.model_json_schema()),
                "strict": True,
            }
        }
        started = time.perf_counter()
        response = await self._client.responses.create(
            model=request.model, instructions=request.system, input=messages, text=text
        )
        latency_ms = (time.perf_counter() - started) * 1000
        usage = response.usage
        return LLMResponse(
            text=response.output_text,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            latency_ms=latency_ms,
        )


def strict_schema(schema: Any) -> Any:
    """OpenAI's strict mode needs every object closed and every property required."""
    if isinstance(schema, dict):
        schema = {key: strict_schema(value) for key, value in schema.items()}
        if schema.get("type") == "object" and "properties" in schema:
            schema["additionalProperties"] = False
            schema["required"] = list(schema["properties"])
        return schema
    if isinstance(schema, list):
        return [strict_schema(item) for item in schema]
    return schema
