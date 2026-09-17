"""Task-general multimodal schedule for the source-constrained discovery accuracy recovery study.

The schedule allocates a fixed ten-fit budget to complementary hypotheses and
then to the modeling axes that matter in any grouped, multi-output response
task.  It never names a dataset, feature, endpoint, or target modality.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlotInstruction:
    slot: int
    mode: str
    goal: str
    max_component_edits: int
    required_components: tuple[str, ...] = ()
    exact_edits: tuple[tuple[str, str], ...] = ()
    parent_policy: str = "incumbent"


SCHEDULE = (
    # Orthogonal anchors make fusion evidence interpretable.  Each remains a
    # complete executable model because all untouched components use the
    # compiler's registered defaults.
    SlotInstruction(1, "complete_hypothesis", "context-anchored perturbation adapter", 1, exact_edits=(("fusion", "context_residual_adapter"),)),
    SlotInstruction(2, "complete_hypothesis", "explicit three-token interaction", 1, exact_edits=(("fusion", "multitoken_attention"),)),
    SlotInstruction(3, "complete_hypothesis", "bilinear perturbation-context interaction", 1, exact_edits=(("fusion", "bilinear_gate"),)),
    SlotInstruction(4, "local_revision", "specialize heterogeneous output groups", 2, ("response_readout", "response_program")),
    # Matched branch expansions prevent a lucky initial seed from eliminating
    # an otherwise useful fusion family before its capacity is tested.
    SlotInstruction(5, "complementary_challenge", "moderate-capacity expansion of anchor A", 1, exact_edits=(("model_architecture", "hidden_384"),), parent_policy="candidate_01"),
    SlotInstruction(6, "local_revision", "moderate-capacity expansion of anchor B", 1, exact_edits=(("model_architecture", "hidden_384"),), parent_policy="candidate_02"),
    SlotInstruction(7, "complementary_challenge", "moderate-capacity expansion of anchor C", 1, exact_edits=(("model_architecture", "hidden_384"),), parent_policy="candidate_03"),
    SlotInstruction(8, "local_revision", "high-capacity challenge of the current evidence leader", 1, exact_edits=(("model_architecture", "hidden_512"),)),
    # Evaluate correlation alignment only after the structural frontier is
    # known, so it cannot mask a useful fusion branch during initialization.
    SlotInstruction(9, "evidence_integration", "correlation-sensitive objective check on the evidence leader", 1, exact_edits=(("loss", "pcc_mse"),)),
)


def instruction_for(slot: int) -> SlotInstruction:
    if slot not in range(1, 10):
        raise ValueError("slot must lie in 1..9")
    return SCHEDULE[slot - 1]
