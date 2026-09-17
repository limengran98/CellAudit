"""Fixed nine-slot diversity-to-convergence schedule for source-constrained discovery."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlotInstruction:
    slot: int
    mode: str
    goal: str
    max_component_edits: int | None


SCHEDULE = (
    SlotInstruction(1, "complete_hypothesis", "complementary full hypothesis A", None),
    SlotInstruction(2, "complete_hypothesis", "complementary full hypothesis B", None),
    SlotInstruction(3, "complete_hypothesis", "complementary full hypothesis C", None),
    SlotInstruction(4, "local_revision", "one- or two-component incumbent revision", 2),
    SlotInstruction(5, "complementary_challenge", "bounded challenger against incumbent", 2),
    SlotInstruction(6, "local_revision", "diagnostic-conditioned incumbent revision", 2),
    SlotInstruction(7, "complementary_challenge", "mechanistically distinct challenger", 2),
    SlotInstruction(8, "local_revision", "diagnostic-conditioned incumbent revision", 2),
    SlotInstruction(9, "evidence_integration", "integrate only Fold-3-positive components", 2),
)


def instruction_for(slot: int) -> SlotInstruction:
    if slot not in range(1, 10):
        raise ValueError("slot must lie in 1..9")
    return SCHEDULE[slot - 1]
