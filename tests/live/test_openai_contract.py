import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel

from evalgate.config import load_config
from evalgate.contracts import LLMRequest, Message
from evalgate.providers.openai import OpenAIClient

pytestmark = pytest.mark.live

EXAMPLE_CONFIG = (
    Path(__file__).parents[2] / "examples" / "support_email_classifier" / "feature.yaml"
)


class Answer(BaseModel):
    city: str
    country_code: str


def test_adapter_returns_raw_structured_output_with_tokens_and_latency() -> None:
    request = LLMRequest(
        model=load_config(EXAMPLE_CONFIG).model,
        system="Answer with the capital city and its ISO 3166-1 alpha-2 country code.",
        messages=(Message(role="user", content="What is the capital of France?"),),
        output_type=Answer,
    )

    response = asyncio.run(OpenAIClient().complete(request))

    answer = Answer.model_validate_json(response.raw_text)
    assert answer.country_code == "FR"
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.latency_ms > 0
