from __future__ import annotations

import os
from typing import Dict, List, TypedDict

from openai import OpenAI

from usage_tracker import record_usage


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

    This function also records token usage and estimated cost per model.
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

    # Extract token usage if available
    input_tokens = 0
    output_tokens = 0

    usage = getattr(response, "usage", None)
    if usage is not None:
        # Newer APIs may use input_tokens/output_tokens,
        # older Chat Completions use prompt_tokens/completion_tokens.
        input_tokens = (
            getattr(usage, "input_tokens", None)
            or getattr(usage, "prompt_tokens", 0)
        )
        output_tokens = (
            getattr(usage, "output_tokens", None)
            or getattr(usage, "completion_tokens", 0)
        )

    record_usage(model=model, input_tokens=input_tokens, output_tokens=output_tokens)

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
