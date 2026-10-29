"""The example feature under test: a customer-support email classifier.

It is wired up exactly as a feature in another repo would be: evalgate finds `feature` through
`feature.yaml` and never imports this module by name.
"""

import json
from typing import Literal

from pydantic import BaseModel

from evalgate.contracts import FeatureUnderTest, Message, PromptVersion, StructuredLLM


class EmailInput(BaseModel):
    subject: str
    body: str


class Classification(BaseModel):
    category: Literal["billing", "technical", "account", "general"]
    summary: str


def _render(email: EmailInput) -> str:
    return f"Subject: {email.subject}\n\n{email.body}"


async def classify(email: EmailInput, prompt: PromptVersion, llm: StructuredLLM) -> Classification:
    messages: list[Message] = []

    for shot in prompt.few_shot:
        shot_input = EmailInput.model_validate(shot.input)

        messages.append(Message(role="user", content=_render(shot_input)))
        messages.append(Message(role="assistant", content=json.dumps(shot.output)))

    messages.append(Message(role="user", content=_render(email)))

    return await llm.generate(
        system=prompt.system_prompt, messages=messages, output_type=Classification
    )


feature = FeatureUnderTest(input_type=EmailInput, output_type=Classification, run=classify)
