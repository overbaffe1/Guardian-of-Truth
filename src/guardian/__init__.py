"""Guardian of Truth: black-box contextual hallucination detector.

The package is intentionally dependency-light at import time: heavy ML
dependencies (torch / vllm / transformers) are imported lazily by the
backends module so that rule-only mode and unit tests work on CPU.
"""

from .types import Event, Prediction, ResponseTurn, Signal, ToolCall, Transcript

__all__ = [
    "Event",
    "Prediction",
    "ResponseTurn",
    "Signal",
    "ToolCall",
    "Transcript",
]

__version__ = "0.3.0"
