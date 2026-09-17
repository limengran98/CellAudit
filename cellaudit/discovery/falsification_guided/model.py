"""Deterministic context-anchor plus perturbation-increment model family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class FalsificationGuidedDesign:
    """Small compiler-owned search language for the residual branch only."""

    fusion: str = "factorized_gate"
    hidden_dim: int = 384
    readout: str = "dual_heads"
    objective: str = "balanced_residual"

    def validate(self) -> None:
        if self.fusion not in {"centered_additive", "factorized_gate", "low_rank_bilinear"}:
            raise ValueError("falsification-guided discovery fusion is not registered")
        if self.hidden_dim not in {256, 384, 512}:
            raise ValueError("falsification-guided discovery hidden width is not registered")
        if self.readout not in {"shared", "dual_heads"}:
            raise ValueError("falsification-guided discovery readout is not registered")
        if self.objective not in {"balanced_residual", "correlation_residual"}:
            raise ValueError("falsification-guided discovery objective is not registered")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "fusion": self.fusion,
            "hidden_dim": self.hidden_dim,
            "readout": self.readout,
            "objective": self.objective,
        }


class _ContextAnchor(nn.Module):
    def __init__(self, context_dim: int, target_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.readout = nn.Linear(hidden_dim, target_dim)

    def forward(self, control: Any) -> tuple[Any, Any]:
        hidden = self.encoder(control)
        return self.readout(hidden), hidden


class _CenteredEncoder(nn.Module):
    """Encode an input relative to a frozen fit-partition reference."""

    def __init__(self, input_dim: int, hidden_dim: int, reference: Any) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        if tuple(reference.shape) != (1, input_dim):
            raise ValueError("falsification-guided discovery centered-encoder reference has the wrong shape")
        self.register_buffer("reference", reference.detach().clone())

    def forward(self, value: Any) -> Any:
        # Center before the network and remove every bias.  This makes a null
        # input exactly zero instead of relying on subtraction of two nearly
        # equal GEMM results whose round-off can depend on batch shape.
        return self.network(value - self.reference)


class PerturbationIncrementModel(nn.Module):
    """A frozen-able context anchor plus a source-identifiable increment.

    The increment has no bias-only route to the output.  Its final hidden state
    is always the product of a context gate and a centered perturbation token.
    The residual readout is zero-initialized, making the initial full predictor
    exactly equal to the context anchor.
    """

    def __init__(
        self,
        *,
        context_dim: int,
        chemical_dim: int,
        dose_dim: int,
        cp_dim: int,
        l1000_dim: int,
        chemical_reference: Any,
        dose_reference: Any,
        design: FalsificationGuidedDesign,
        anchor_hidden_dim: int = 384,
    ) -> None:
        super().__init__()
        design.validate()
        target_dim = int(cp_dim + l1000_dim)
        hidden = int(design.hidden_dim)
        self.design = design
        self.cp_dim = int(cp_dim)
        self.register_buffer("increment_scale", torch.ones((), dtype=torch.float32))
        self.anchor = _ContextAnchor(int(context_dim), target_dim, int(anchor_hidden_dim))
        self.context_gate = nn.Sequential(nn.Linear(int(anchor_hidden_dim), hidden), nn.Sigmoid())
        self.chemical_encoder = _CenteredEncoder(int(chemical_dim), hidden, chemical_reference)
        self.dose_encoder = _CenteredEncoder(int(dose_dim), hidden, dose_reference)
        self.chemical_projection = nn.Linear(hidden, hidden, bias=False)
        self.dose_projection = nn.Linear(hidden, hidden, bias=False)
        if design.fusion == "low_rank_bilinear":
            rank = min(64, hidden)
            self.chemical_rank = nn.Linear(hidden, rank, bias=False)
            self.dose_rank = nn.Linear(hidden, rank, bias=False)
            self.interaction_up = nn.Linear(rank, hidden, bias=False)
        else:
            self.chemical_rank = None
            self.dose_rank = None
            self.interaction_up = None
        self.chemical_program = nn.Sequential(
            nn.Linear(hidden, hidden, bias=False),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.dose_program = nn.Sequential(
            nn.Linear(hidden, hidden, bias=False),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        if design.readout == "shared":
            self.chemical_readout = nn.Linear(hidden, target_dim, bias=False)
            self.dose_readout = nn.Linear(hidden, target_dim, bias=False)
            self.chemical_cp_readout = None
            self.chemical_l1000_readout = None
            self.dose_cp_readout = None
            self.dose_l1000_readout = None
        else:
            self.chemical_readout = None
            self.dose_readout = None
            self.chemical_cp_readout = nn.Linear(hidden, int(cp_dim), bias=False)
            self.chemical_l1000_readout = nn.Linear(hidden, int(l1000_dim), bias=False)
            self.dose_cp_readout = nn.Linear(hidden, int(cp_dim), bias=False)
            self.dose_l1000_readout = nn.Linear(hidden, int(l1000_dim), bias=False)
        self.reset_increment()

    def reset_increment(self) -> None:
        readouts = [
            self.chemical_readout, self.dose_readout,
            self.chemical_cp_readout, self.chemical_l1000_readout,
            self.dose_cp_readout, self.dose_l1000_readout,
        ]
        for layer in readouts:
            if layer is not None:
                nn.init.zeros_(layer.weight)

    def anchor_prediction(self, control: Any) -> Any:
        prediction, _hidden = self.anchor(control)
        return prediction

    @staticmethod
    def _apply_readout(shared: Any, cp: Any, l1000: Any, hidden: Any) -> Any:
        if shared is not None:
            return shared(hidden)
        if cp is None or l1000 is None:
            raise RuntimeError("falsification-guided discovery dual readout is incomplete")
        return torch.cat((cp(hidden), l1000(hidden)), dim=1)

    def perturbation_components(self, control: Any, chemical: Any, dose: Any) -> tuple[Any, Any]:
        _anchor, context = self.anchor(control)
        chemical_token = self.chemical_projection(self.chemical_encoder(chemical))
        dose_token = self.dose_projection(self.dose_encoder(dose))
        gate = self.context_gate(context)
        chemical_hidden = self.chemical_program(gate * chemical_token)
        if self.design.fusion == "centered_additive":
            dose_fused = dose_token
        elif self.design.fusion == "factorized_gate":
            dose_fused = dose_token * (1.0 + torch.tanh(chemical_token))
        else:
            if self.chemical_rank is None or self.dose_rank is None or self.interaction_up is None:
                raise RuntimeError("falsification-guided discovery bilinear modules are absent")
            interaction = self.interaction_up(
                self.chemical_rank(chemical_token) * self.dose_rank(dose_token)
            )
            dose_fused = dose_token + interaction
        dose_hidden = self.dose_program(gate * dose_fused)
        chemical_increment = self._apply_readout(
            self.chemical_readout, self.chemical_cp_readout, self.chemical_l1000_readout, chemical_hidden
        )
        dose_increment = self._apply_readout(
            self.dose_readout, self.dose_cp_readout, self.dose_l1000_readout, dose_hidden
        )
        return self.increment_scale * chemical_increment, self.increment_scale * dose_increment

    def perturbation_increment(self, control: Any, chemical: Any, dose: Any) -> Any:
        chemical_increment, dose_increment = self.perturbation_components(control, chemical, dose)
        return chemical_increment + dose_increment

    def forward_components(self, control: Any, chemical: Any, dose: Any) -> tuple[Any, Any, Any, Any]:
        anchor = self.anchor_prediction(control)
        chemical_increment, dose_increment = self.perturbation_components(control, chemical, dose)
        return anchor + chemical_increment + dose_increment, anchor, chemical_increment, dose_increment

    def forward_parts(self, control: Any, chemical: Any, dose: Any) -> tuple[Any, Any, Any]:
        full, anchor, chemical_increment, dose_increment = self.forward_components(control, chemical, dose)
        return full, anchor, chemical_increment + dose_increment

    def forward(self, control: Any, chemical: Any, dose: Any) -> Any:
        return self.forward_parts(control, chemical, dose)[0]

    def freeze_anchor(self) -> None:
        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)

    def set_increment_scale(self, value: float) -> None:
        scale = float(value)
        if not 0.0 <= scale <= 2.0:
            raise ValueError("falsification-guided discovery increment scale is outside the registered safe range")
        self.increment_scale.fill_(scale)

    def anchor_parameters(self):
        return self.anchor.parameters()

    def increment_parameters(self):
        return (parameter for name, parameter in self.named_parameters() if not name.startswith("anchor."))

    def increment_readout_parameters(self):
        return (
            parameter for name, parameter in self.named_parameters()
            if not name.startswith("anchor.") and "readout" in name
        )

    def increment_feature_parameters(self):
        return (
            parameter for name, parameter in self.named_parameters()
            if not name.startswith("anchor.") and "readout" not in name
        )
