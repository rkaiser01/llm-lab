from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelUsage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


# Prices per 1M tokens (approximate, based on OpenAI pricing docs as of 2025)
# See: https://platform.openai.com/docs/pricing
MODEL_PRICES_PER_MILLION: Dict[str, Dict[str, float]] = {
    "gpt-4.1-mini": {
        "input": 0.40,   # $ per 1M input tokens
        "output": 1.60,  # $ per 1M output tokens
    },
    "gpt-4.1": {
        "input": 2.00,   # $ per 1M input tokens
        "output": 8.00,  # $ per 1M output tokens
    },
    "o3-mini": {
        "input": 1.10,   # $ per 1M input tokens
        "output": 4.40,  # $ per 1M output tokens
    },
}

_model_usage: Dict[str, ModelUsage] = {}


def reset_usage() -> None:
    """
    Reset all tracked usage data.
    This is effectively called once per process run if imported at startup.
    """
    _model_usage.clear()


def _get_or_create_usage(model: str) -> ModelUsage:
    if model not in _model_usage:
        _model_usage[model] = ModelUsage()
    return _model_usage[model]


def _calculate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate the cost in USD for the given token usage and model.

    If no pricing information is available for the model, returns 0.0.
    """
    prices = MODEL_PRICES_PER_MILLION.get(model)
    if not prices:
        return 0.0

    input_price = prices["input"]
    output_price = prices["output"]

    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price

    return input_cost + output_cost


def record_usage(model: str, input_tokens: int, output_tokens: int) -> None:
    """
    Record token usage for a given model and update the estimated cost.
    """
    if input_tokens < 0:
        input_tokens = 0
    if output_tokens < 0:
        output_tokens = 0

    usage = _get_or_create_usage(model)
    usage.requests += 1
    usage.input_tokens += input_tokens
    usage.output_tokens += output_tokens

    cost = _calculate_cost_usd(model, input_tokens, output_tokens)
    usage.cost_usd += cost


def get_usage_summary() -> Dict[str, ModelUsage]:
    """
    Return a shallow copy of the internal usage data.
    """
    return dict(_model_usage)


def format_usage_summary() -> str:
    """
    Return a human-readable summary of usage and cost for printing in the console.
    """
    if not _model_usage:
        return "Usage summary: no tokens used yet."

    lines = ["Usage summary (per model):"]
    total_input = 0
    total_output = 0
    total_cost = 0.0

    for model, usage in _model_usage.items():
        lines.append(
            f"  - {model}: requests={usage.requests}, "
            f"input_tokens={usage.input_tokens}, "
            f"output_tokens={usage.output_tokens}, "
            f"estimated_cost=${usage.cost_usd:.6f}"
        )
        total_input += usage.input_tokens
        total_output += usage.output_tokens
        total_cost += usage.cost_usd

    lines.append(
        f"Total: input_tokens={total_input}, "
        f"output_tokens={total_output}, "
        f"estimated_cost=${total_cost:.6f}"
    )

    return "\n".join(lines)
