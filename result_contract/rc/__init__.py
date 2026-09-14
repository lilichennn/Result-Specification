"""Result Contract generation."""

from .rc_round1 import MODEL_ALIASES, ModelCall, Round1RC, call_model, generate_round1
from .rc_round2 import Round2RC, generate_round2
from .rc_round3 import Round3RC, generate_round3

__all__ = [
    "MODEL_ALIASES",
    "ModelCall",
    "Round1RC",
    "Round2RC",
    "Round3RC",
    "call_model",
    "generate_round1",
    "generate_round2",
    "generate_round3",
]
