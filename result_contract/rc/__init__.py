"""Result Contract generation."""

from .rc_round1 import ModelCall, Round1RC, call_model, generate_round1
from .rc_round2 import Round2RC, generate_round2

__all__ = [
    "ModelCall",
    "Round1RC",
    "Round2RC",
    "call_model",
    "generate_round1",
    "generate_round2",
]
