from __future__ import annotations

from typing import Tuple

from llm_client import (
    chat_with_messages,
    ChatMessage,
    get_default_model,
)


def advisor_step(
    user_prompt: str,
    advisor_model: str = "o3-mini",
) -> str:
    """
    Let a dedicated advisor model analyze the user prompt
    and suggest how the main model should answer.

    The advisor should NOT answer the user directly, but provide
    guidance, structure, and potential pitfalls.
    """
    advisor_system = (
        "You are an AI advisor for another LLM. "
        "Your task is to analyze the user's question and provide clear, "
        "concise guidance for how another assistant should answer. "
        "Do NOT answer the user directly. "
        "Focus on: key steps, structure, and important considerations."
    )

    messages: list[ChatMessage] = [
        {"role": "system", "content": advisor_system},
        {"role": "user", "content": user_prompt},
    ]

    advisor_notes = chat_with_messages(
        messages=messages,
        model=advisor_model,
        max_tokens=512,
    )
    return advisor_notes


def primary_step(
    user_prompt: str,
    advisor_notes: str,
    primary_model: str | None = None,
) -> str:
    """
    Let the primary model answer the user, taking the advisor notes into account.
    """
    if primary_model is None:
        primary_model = get_default_model()

    primary_system = (
        "You are the primary assistant answering the user. "
        "Another AI advisor has provided you with analysis and guidance. "
        "Carefully read and use these advisor notes to produce a high-quality answer. "
        "Do NOT mention the advisor or that you were advised. "
        "Just provide the best possible answer to the user."
    )

    messages: list[ChatMessage] = [
        {"role": "system", "content": primary_system},
        {
            "role": "user",
            "content": (
                "User question:\n"
                f"{user_prompt}\n\n"
                "Advisor notes (for you, not for the user):\n"
                f"{advisor_notes}"
            ),
        },
    ]

    final_answer = chat_with_messages(
        messages=messages,
        model=primary_model,
        max_tokens=512,
    )
    return final_answer


def orchestrated_chat(
    user_prompt: str,
    primary_model: str | None = None,
    advisor_model: str = "o3-mini",
    return_advisor: bool = False,
) -> str | tuple[str, str]:
    """
    Full orchestration:
    1) Ask the advisor model for guidance.
    2) Use the guidance to generate the final answer with the primary model.

    If return_advisor is True, returns (final_answer, advisor_notes),
    otherwise only the final answer.
    """
    advisor_notes = advisor_step(user_prompt, advisor_model=advisor_model)
    final_answer = primary_step(
        user_prompt=user_prompt,
        advisor_notes=advisor_notes,
        primary_model=primary_model,
    )

    if return_advisor:
        return final_answer, advisor_notes

    return final_answer
