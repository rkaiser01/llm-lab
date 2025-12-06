from __future__ import annotations

from llm_client import AVAILABLE_MODELS, get_default_model
from orchestrator import orchestrated_chat

from usage_tracker import format_usage_summary


def choose_model() -> str:
    """
    Ask the user to choose a primary model from AVAILABLE_MODELS.
    """
    print("Available primary models:")
    for idx, (model_name, description) in enumerate(AVAILABLE_MODELS.items(), start=1):
        default_marker = " (default)" if model_name == get_default_model() else ""
        print(f"  {idx}. {model_name}{default_marker} - {description}")

    print()
    raw_choice = input(
        f"Choose a primary model by number or press Enter for default [{get_default_model()}]: "
    ).strip()

    if not raw_choice:
        chosen_model = get_default_model()
        print(f"Using default model: {chosen_model}")
        return chosen_model

    try:
        index = int(raw_choice)
    except ValueError:
        print("Invalid input. Using default model.")
        return get_default_model()

    model_list = list(AVAILABLE_MODELS.keys())
    if index < 1 or index > len(model_list):
        print("Number out of range. Using default model.")
        return get_default_model()

    chosen_model = model_list[index - 1]
    print(f"Using primary model: {chosen_model}")
    return chosen_model


def main() -> None:
    """
    Simple CLI loop to chat with the LLM in the terminal.

    For each user prompt, an advisor model is consulted first,
    then the primary model generates the final answer.
    """
    print("LLM Lab - orchestrated chat CLI")
    print("--------------------------------\n")

    primary_model = choose_model()
    advisor_model = "o3-mini"  # fixed for now; can also be made interactive

    print(f"Advisor model (fixed): {advisor_model}")
    print("\nType 'exit' to quit.\n")

    while True:
        user_prompt = input("You: ").strip()
        if not user_prompt:
            continue
        if user_prompt.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        try:
            final_answer, advisor_notes = orchestrated_chat(
                user_prompt=user_prompt,
                primary_model=primary_model,
                advisor_model=advisor_model,
                return_advisor=True,
            )
        except Exception as exc:
            print(f"Error while calling LLMs: {exc}")
            continue

        # Show intermediate advisor step
        print("\n[Advisor notes]")
        print(advisor_notes)

        # Show final answer
        print("\nAssistant:")
        print(final_answer)

        # Show aggregated usage and cost
        print("\n[Usage]")
        print(format_usage_summary())
        print()


if __name__ == "__main__":
    main()
