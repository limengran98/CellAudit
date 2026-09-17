"""Schema-first objects for the source-constrained discovery open-discovery controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


EDITABLE_COMPONENTS = frozenset({
    "model_architecture", "chemical_encoder", "fusion", "response_program",
    "response_readout", "loss", "optimizer", "scheduler",
})


class DesignCardError(ValueError):
    pass


@dataclass(frozen=True)
class ComponentEdit:
    component: str
    mechanism: str
    rationale: str

    def validate(self) -> None:
        if self.component not in EDITABLE_COMPONENTS:
            raise DesignCardError(f"unregistered editable component: {self.component}")
        if not self.mechanism.strip() or not self.rationale.strip():
            raise DesignCardError("each edit needs a mechanism and a falsifiable rationale")


@dataclass(frozen=True)
class DesignCard:
    candidate_id: str
    slot: int
    parent_candidate_id: str | None
    mode: str
    hypothesis: str
    edits: tuple[ComponentEdit, ...]
    evidence_candidate_ids: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "DesignCard":
        raw_edits = value.get("edits")
        if not isinstance(raw_edits, Sequence) or isinstance(raw_edits, (str, bytes)):
            raise DesignCardError("edits must be a list")
        edits: list[ComponentEdit] = []
        for item in raw_edits:
            if not isinstance(item, Mapping):
                raise DesignCardError("every edit must be an object")
            edits.append(ComponentEdit(
                component=str(item.get("component", "")),
                mechanism=str(item.get("mechanism", "")),
                rationale=str(item.get("rationale", "")),
            ))
        evidence = value.get("evidence_candidate_ids", ())
        if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
            raise DesignCardError("evidence_candidate_ids must be a list")
        card = cls(
            candidate_id=str(value.get("candidate_id", "")),
            slot=int(value.get("slot", 0)),
            parent_candidate_id=(str(value["parent_candidate_id"]) if value.get("parent_candidate_id") else None),
            mode=str(value.get("mode", "")),
            hypothesis=str(value.get("hypothesis", "")),
            edits=tuple(edits),
            evidence_candidate_ids=tuple(str(item) for item in evidence),
        )
        card.validate()
        return card

    def validate(self) -> None:
        if not self.candidate_id or self.slot not in range(1, 10):
            raise DesignCardError("candidate_id and a slot in 1..9 are required")
        if self.mode not in {"complete_hypothesis", "local_revision", "complementary_challenge", "evidence_integration"}:
            raise DesignCardError("design-card mode is invalid")
        if not self.hypothesis.strip() or not self.edits:
            raise DesignCardError("a hypothesis and at least one edit are required")
        if len({edit.component for edit in self.edits}) != len(self.edits):
            raise DesignCardError("each semantic component may be edited once per slot")
        for edit in self.edits:
            edit.validate()
        if self.slot <= 3:
            if self.mode != "complete_hypothesis" or self.parent_candidate_id is not None:
                raise DesignCardError("slots 1–3 require independent complete hypotheses")
        elif self.slot <= 8:
            if self.mode not in {"local_revision", "complementary_challenge"} or not self.parent_candidate_id:
                raise DesignCardError("slots 4–8 require a parent and a bounded revision/challenge")
            if len(self.edits) > 2:
                raise DesignCardError("slots 4–8 may change at most two semantic components")
        else:
            if self.mode != "evidence_integration" or not self.evidence_candidate_ids:
                raise DesignCardError("slot 9 requires evidence-backed integration")
            if len(self.edits) > 2:
                raise DesignCardError("slot 9 may integrate at most two supported components")
