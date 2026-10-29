import time
from typing import Any

from openai import AsyncOpenAI
from openai.types.responses import ResponseInputParam, ResponseTextConfigParam

from evalgate.contracts import LLMRequest, LLMResponse


class OpenAIClient:
    def __init__(self) -> None:
        self._client = AsyncOpenAI()

    async def complete(self, request: LLMRequest) -> LLMResponse:
        messages: ResponseInputParam = [
            {"role": message.role, "content": message.content} for message in request.messages
        ]

        output_format: ResponseTextConfigParam = {
            "format": {
                "type": "json_schema",
                "name": request.output_type.__name__,
                "schema": to_strict_json_schema(request.output_type.model_json_schema()),
                "strict": True,
            }
        }

        started_at = time.perf_counter()
        response = await self._client.responses.create(
            model=request.model, instructions=request.system, input=messages, text=output_format
        )
        latency_ms = (time.perf_counter() - started_at) * 1000

        usage = response.usage

        return LLMResponse(
            raw_text=response.output_text,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            latency_ms=latency_ms,
        )


def to_strict_json_schema(schema: Any) -> Any:
    # OpenAI's strict mode rejects any object that is open or has an optional property.
    if isinstance(schema, dict):
        schema = {key: to_strict_json_schema(value) for key, value in schema.items()}
        if schema.get("type") == "object" and "properties" in schema:
            schema["additionalProperties"] = False
            schema["required"] = list(schema["properties"])

        return schema

    if isinstance(schema, list):
        return [to_strict_json_schema(item) for item in schema]

    return schema
