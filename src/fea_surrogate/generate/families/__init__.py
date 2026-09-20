"""Component families. Each supplies a sampler, a builder and a tagging rule."""

from __future__ import annotations

from .bracket import Bracket
from .flange import Flange
from .pulley import Pulley
from .shaft import SteppedShaft
from .spring import Spring
from .welded_t import WeldedT


ALL_FAMILIES = (Bracket, Flange, Pulley, SteppedShaft, Spring, WeldedT)

__all__ = [
    "ALL_FAMILIES",
    "Bracket",
    "Flange",
    "Pulley",
    "Spring",
    "SteppedShaft",
    "WeldedT",
]
