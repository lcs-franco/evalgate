import json
from typing import Any

from evalgate.contracts import LLMRequest, LLMResponse


def reply(category: str, summary: str = "A summary.") -> str:
    return json.dumps({"category": category, "summary": summary})


class ScriptedLLM:
    """Returns a fixed raw output for each case, picked by a key found in the case's message."""

    def __init__(self, script: dict[str, str], *, tokens: tuple[int, int] = (120, 30)) -> None:
        self._script = script
        self._tokens = tokens

    async def complete(self, request: LLMRequest) -> LLMResponse:
        last_message = request.messages[-1].content
        matches = [text for key, text in self._script.items() if key in last_message]
        if len(matches) != 1:
            raise AssertionError(f"script has {len(matches)} replies for: {last_message!r}")
        return LLMResponse(
            text=matches[0],
            input_tokens=self._tokens[0],
            output_tokens=self._tokens[1],
            latency_ms=42.0,
        )


def case(
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
