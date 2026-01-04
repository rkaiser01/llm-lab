from __future__ import annotations

from llm_client import AVAILABLE_MODELS, get_default_model
from orchestrator import orchestrated_chat
from usage_tracker import format_usage_summary
from datetime import datetime


def ts() -> str:
    """
    Return a formatted timestamp for console output.
    """
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def choose_model() -> str:
    print(f"{ts()} SYSTEM > Available primary models:")
    for idx, (model_name, desc) in enumerate(AVAILABLE_MODELS.items(), start=1):
        default_mark = " (default)" if model_name == get_default_model() else ""
        print(f"  {idx}. {model_name}{default_mark} - {desc}")

    raw = input(f"{ts()} SYSTEM > Choose model (Enter = default): ").strip()

    if not raw:
        chosen = get_default_model()
        print(f"{ts()} SYSTEM > Using default model: {chosen}")
        return chosen

    try:
        idx = int(raw)
        models = list(AVAILABLE_MODELS.keys())
        if 1 <= idx <= len(models):
            chosen = models[idx - 1]
            print(f"{ts()} SYSTEM > Using model: {chosen}")
            return chosen
    except ValueError:
        pass

    chosen = get_default_model()
    print(f"{ts()} SYSTEM > Invalid choice. Using default: {chosen}")
    return chosen


def main() -> None:
    print(f"{ts()} SYSTEM > LLM Lab orchestrated CLI started")
    primary_model = choose_model()
    advisor_model = "o3-mini"

    print(f"{ts()} SYSTEM > Advisor model: {advisor_model}")
    print(f"{ts()} SYSTEM > Type 'exit' to quit.\n")

    while True:
        user_prompt = input("YOU > ").strip()

        if not user_prompt:
            continue
        if user_prompt.lower() in {"exit", "quit"}:
            print(f"{ts()} SYSTEM > Exiting application.")
            break

        print(f"{ts()} USER INPUT > {user_prompt}")

        try:
            final_answer, advisor_notes = orchestrated_chat(
                user_prompt=user_prompt,
                primary_model=primary_model,
                advisor_model=advisor_model,
                return_advisor=True,
            )
        except Exception as exc:
            print(f"{ts()} ERROR > {exc}")
            continue

        # Advisor step
        print(f"\n{ts()} ADVISOR > {advisor_notes}\n")

        # Primary answer
        print(f"{ts()} ASSISTANT > {final_answer}\n")

        # Usage summary
        print(f"{ts()} USAGE >")
        print(format_usage_summary())
        print()


if __name__ == "__main__":
    main()
