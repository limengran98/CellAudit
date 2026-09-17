"""Fold-3-only, stability-aware deterministic endpoint tie-breaking."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean, stdev
from typing import Sequence


@dataclass(frozen=True)
class Fold3CandidateScore:
    candidate_id: str
    global_pcc: float
    mse: float
    block_pcc: tuple[float, ...]

    @property
    def block_lcb(self) -> float:
        if len(self.block_pcc) < 2:
            raise ValueError("at least two pre-registered Fold-3 blocks are required")
        return mean(self.block_pcc) - 1.96 * stdev(self.block_pcc) / math.sqrt(len(self.block_pcc))


def select_endpoint(candidates: Sequence[Fold3CandidateScore]) -> Fold3CandidateScore:
    if not candidates:
        raise ValueError("at least one Fold-3 candidate is required")
    return sorted(candidates, key=lambda item: (-item.global_pcc, item.mse, -item.block_lcb, item.candidate_id))[0]
