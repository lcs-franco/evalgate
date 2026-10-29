import json
from typing import Any

from evalgate.contracts import LLMRequest, LLMResponse


def model_reply(category: str, summary: str = "A summary.") -> str:
    return json.dumps({"category": category, "summary": summary})


class ScriptedLLM:
    def __init__(
        self,
        replies_by_message_snippet: dict[str, str],
        *,
        input_tokens: int = 120,
        output_tokens: int = 30,
    ) -> None:
        self._replies_by_message_snippet = replies_by_message_snippet
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens

    async def complete(self, request: LLMRequest) -> LLMResponse:
        case_message = request.messages[-1].content
        matching_replies = [
            reply
            for snippet, reply in self._replies_by_message_snippet.items()
            if snippet in case_message
        ]

        if len(matching_replies) != 1:
            raise AssertionError(f"{len(matching_replies)} scripted replies for: {case_message!r}")

        return LLMResponse(
            raw_text=matching_replies[0],
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            latency_ms=42.0,
        )


def golden_case(
    case_id: str, subject: str, category: str, *, summary: str = "A summary.", **overrides: Any
) -> dict[str, Any]:
    return {
        "id": case_id,
        "input": {"subject": subject, "body": f"Body of {subject}."},
        "expected": {"category": category, "summary": summary},
        "expected_difficulty": "easy",
        "why": "exercises the pipeline",
        **overrides,
    }
