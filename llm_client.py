from __future__ import annotations

import os
from typing import Dict, List, TypedDict

from openai import OpenAI


class ChatMessage(TypedDict):
    role: str
    content: str


AVAILABLE_MODELS: Dict[str, str] = {
    "gpt-4.1-mini": "Fast and cheap, good for quick prototypes.",
    "gpt-4.1": "Stronger reasoning, slightly slower and more expensive.",
    "o3-mini": "High reasoning performance for complex tasks.",
}


def get_default_model() -> str:
    """
    Return the default model identifier used when none is explicitly chosen.
    """
    return "gpt-4.1-mini"


def get_openai_client() -> OpenAI:
    """
    Create and return an OpenAI client instance.

    Expects the API key to be available in the OPENAI_API_KEY environment variable.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Configure it in a .env file or in your run configuration."
        )

    return OpenAI()


def validate_model(model: str) -> str:
    """
    Validate that the given model is known and supported.

    Raises a ValueError if the model is not in AVAILABLE_MODELS.
    """
    if model not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model '{model}'. "
            f"Supported models are: {', '.join(AVAILABLE_MODELS.keys())}"
        )
    return model


def chat_with_messages(
    messages: List[ChatMessage],
    model: str | None = None,
    max_tokens: int = 256,
) -> str:
    """
    Send a list of chat messages to the given model and return the response text.
    """
    client = get_openai_client()

    if model is None:
        model = get_default_model()

    model = validate_model(model)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_tokens,
    )

    message = response.choices[0].message.content
    return message or ""


def simple_chat(prompt: str, model: str | None = None) -> str:
    """
    Convenience wrapper for a simple one-shot user prompt.
    """
    messages: List[ChatMessage] = [
        {
            "role": "system",
            "content": "You are a helpful assistant for quick LLM prototypes.",
        },
        {"role": "user", "content": prompt},
    ]
    return chat_with_messages(messages, model=model)
