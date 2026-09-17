"""Deterministic preflight compiler for structured source-constrained discovery design cards."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping

from .schedule import instruction_for
from .types import DesignCard, DesignCardError


@dataclass(frozen=True)
class CompiledDesign:
    candidate_id: str
    slot: int
    parent_candidate_id: str | None
    shape_contract: Mapping[str, int | str]
    metadata: Mapping[str, object]
    digest: str


_FUSIONS = frozenset({"film", "gated_residual", "multitoken_attention", "bilinear_gate"})
_READOUTS = frozenset({"shared", "dual_heads"})
_HIDDEN = frozenset({256, 384, 512})
_LOSSES = frozenset({"mse", "pcc_mse", "block_balanced_mse"})


def _default_choices() -> dict[str, object]:
    return {
        "fusion": "gated_residual", "readout": "shared", "hidden_dim": 256,
        "dropout": 0.1, "optimizer": "adamw", "scheduler": "cosine", "loss": "mse",
        "bottleneck": False, "deep_chemical_encoder": False, "residual_response": False,
    }


def _choices(card: DesignCard, *, base_choices: Mapping[str, object] | None = None) -> dict[str, object]:
    """Interpret a small, compiler-owned vocabulary from a free hypothesis card."""
    defaults = _default_choices()
    if base_choices is None:
        values = defaults
    else:
        if set(base_choices) != set(defaults):
            raise DesignCardError("parent compiler choices do not match the frozen candidate language")
        values = dict(base_choices)
    for edit in card.edits:
        token = edit.mechanism.strip().lower().replace("-", "_")
        if edit.component == "fusion":
            if token not in _FUSIONS:
                raise DesignCardError("fusion must be film, gated_residual, multitoken_attention, or bilinear_gate")
            values["fusion"] = token
        elif edit.component == "response_readout":
            if token not in _READOUTS:
                raise DesignCardError("response_readout must be shared or dual_heads")
            values["readout"] = token
        elif edit.component == "model_architecture":
            if token.startswith("hidden_") and token.removeprefix("hidden_").isdigit():
                width = int(token.removeprefix("hidden_"))
                if width not in _HIDDEN:
                    raise DesignCardError("hidden width is not compiler-registered")
                values["hidden_dim"] = width
            else:
                raise DesignCardError("model_architecture must be hidden_256/384/512")
        elif edit.component == "chemical_encoder":
            if token == "deep_two_layer":
                values["deep_chemical_encoder"] = True
            elif token != "standard":
                raise DesignCardError("chemical_encoder must be standard or deep_two_layer")
        elif edit.component == "response_program":
            if token == "bottleneck_64":
                values["bottleneck"] = True
            elif token == "residual":
                values["residual_response"] = True
            elif token != "standard":
                raise DesignCardError("response_program must be standard, bottleneck_64, or residual")
        elif edit.component == "loss":
            if token not in _LOSSES:
                raise DesignCardError("loss must be mse, pcc_mse, or block_balanced_mse")
            values["loss"] = token
        elif edit.component == "optimizer":
            if token not in {"adam", "adamw"}:
                raise DesignCardError("optimizer must be adam or adamw")
            values["optimizer"] = token
        elif edit.component == "scheduler":
            if token not in {"cosine", "none"}:
                raise DesignCardError("scheduler must be cosine or none")
            values["scheduler"] = token
    return values


def _mechanism_certificate(*, fusion: str, choices: Mapping[str, object]) -> dict[str, object]:
    """A deterministic source-level contract, not an empirical claim of use."""
    fusion_paths = {
        "film": "chemical/dose modulation of the control representation",
        "gated_residual": "chemical/dose residual gated against the control representation",
        "bilinear_gate": "control-conditioned gate applied to the chemical representation",
        "multitoken_attention": "three-token control/chemical/dose attention; no single-token normalization",
    }
    return {
        "schema_version": "cellscientist_source_constrained_mechanism_certificate_v1",
        "declared_input_contract": "chemical_control_conditioned_response",
        "chemical_path": fusion_paths[fusion],
        "single_token_attention_forbidden": True,
        "attention_token_count": 3 if fusion == "multitoken_attention" else 0,
        "chemical_encoder": "deep_two_layer" if bool(choices["deep_chemical_encoder"]) else "standard",
        "response_program": (
            "bottleneck_64" if bool(choices["bottleneck"])
            else "residual" if bool(choices["residual_response"])
            else "standard"
        ),
        "endpoint_heldout_behavioral_audit_required": True,
    }


def _metadata(*, card: DesignCard, fusion: str, readout: str, choices: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": "cellscientist_open_candidate_v1",
        "candidate_name": f"candidate_{card.candidate_id}_{fusion}_{readout}",
        "semantic_components": [
            "input.control_encoding", "input.chemical_encoding", "input.dose_encoding",
            f"perturbation.{fusion}", "response.shared_program", f"response.{readout}",
            "objective.response_loss", "optimization.optimizer", "optimization.scheduler",
        ],
        "component_symbols": {
            "input.control_encoding": ["ControlEncoder"], "input.chemical_encoding": ["ChemicalEncoder"],
            "input.dose_encoding": ["DoseEncoder"], f"perturbation.{fusion}": ["CompiledMultimodalModel"],
            "response.shared_program": ["CompiledMultimodalModel"], f"response.{readout}": ["CompiledMultimodalModel"],
            "objective.response_loss": ["compute_loss"], "optimization.optimizer": ["build_optimizer"],
            "optimization.scheduler": ["build_scheduler"],
        },
        "change_summary": f"source-constrained discovery compiler materialization of card {card.candidate_id}: {card.hypothesis}",
        "mechanism_certificate": _mechanism_certificate(fusion=fusion, choices=choices),
    }


def _candidate_source(*, card: DesignCard, choices: Mapping[str, object]) -> str:
    """Generate the entire executable model and metadata from a validated card.

    No LLM-provided source, metadata, symbol map, or tensor reshaping reaches
    the executor.  The template always consumes (control, chemical, dose) and
    emits a joint [batch, target_dim] response.
    """
    fusion = str(choices["fusion"])
    readout = str(choices["readout"])
    hidden = int(choices["hidden_dim"])
    dropout = float(choices["dropout"])
    optimizer = str(choices["optimizer"])
    scheduler = str(choices["scheduler"])
    loss_name = str(choices["loss"])
    bottleneck = bool(choices["bottleneck"])
    deep_chemical_encoder = bool(choices["deep_chemical_encoder"])
    residual_response = bool(choices["residual_response"])
    fusion_setup = {
        "film": "self.modulator = nn.Linear(hidden_dim * 2, hidden_dim * 2)",
        "gated_residual": "self.gate = nn.Linear(hidden_dim * 2, hidden_dim)",
        "bilinear_gate": "self.chemical_gate = nn.Linear(hidden_dim, hidden_dim)",
        "multitoken_attention": "self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)",
    }[fusion]
    fusion_forward = {
        "film": "params = self.modulator(torch.cat((chem, dose_h), dim=1)); scale, bias = params.chunk(2, dim=1); fused = ctrl * (1.0 + torch.tanh(scale)) + bias",
        "gated_residual": "gate = torch.sigmoid(self.gate(torch.cat((chem, dose_h), dim=1))); fused = ctrl + gate * (chem + dose_h - ctrl)",
        "bilinear_gate": "fused = ctrl + dose_h + chem * torch.sigmoid(self.chemical_gate(ctrl))",
        "multitoken_attention": "tokens = torch.stack((ctrl, chem, dose_h), dim=1); attended, _ = self.attention(tokens, tokens, tokens, need_weights=False); fused = attended[:, 0, :] + tokens[:, 0, :]",
    }[fusion]
    bottleneck_setup = "self.bottleneck = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, hidden_dim))" if bottleneck else "self.bottleneck = nn.Identity()"
    chemical_encoder_setup = (
        "self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())"
        if deep_chemical_encoder
        else "self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())"
    )
    readout_setup = "self.readout = nn.Linear(hidden_dim, target_dim)" if readout == "shared" else "self.cp_readout = nn.Linear(hidden_dim, cp_dim); self.l1000_readout = nn.Linear(hidden_dim, l1000_dim)"
    readout_forward = "return self.readout(hidden)" if readout == "shared" else "return torch.cat((self.cp_readout(hidden), self.l1000_readout(hidden)), dim=1)"
    optimizer_line = "torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)" if optimizer == "adamw" else "torch.optim.Adam(model.parameters(), lr=1e-3)"
    scheduler_line = "return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(state['max_epochs']))" if scheduler == "cosine" else "return None"
    response_line = "hidden = self.response(self.bottleneck(fused)); hidden = hidden + fused" if residual_response else "hidden = self.response(self.bottleneck(fused))"
    loss_source = {
        "mse": "return torch.nn.functional.mse_loss(prediction, target)",
        "pcc_mse": "mse = torch.nn.functional.mse_loss(prediction, target); pred = prediction - prediction.mean(dim=1, keepdim=True); truth = target - target.mean(dim=1, keepdim=True); corr = (pred * truth).sum(dim=1) / (pred.square().sum(dim=1).sqrt() * truth.square().sum(dim=1).sqrt() + 1e-8); return mse + 0.1 * (1.0 - corr.mean())",
        "block_balanced_mse": "cp_dim = int(state.get('cp_dim', prediction.shape[1] // 2)); return 0.5 * torch.nn.functional.mse_loss(prediction[:, :cp_dim], target[:, :cp_dim]) + 0.5 * torch.nn.functional.mse_loss(prediction[:, cp_dim:], target[:, cp_dim:])",
    }[loss_name]
    metadata = _metadata(card=card, fusion=fusion, readout=readout, choices=choices)
    return f'''import torch
from torch import nn

def candidate_metadata():
    return {metadata!r}

class ControlEncoder(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__(); self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
    def forward(self, value): return self.net(value)

class ChemicalEncoder(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__(); {chemical_encoder_setup}
    def forward(self, value): return self.net(value)

class DoseEncoder(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__(); self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU())
    def forward(self, value): return self.net(value)

class CompiledMultimodalModel(nn.Module):
    def __init__(self, task_spec):
        super().__init__()
        hidden_dim = {hidden}
        target_dim, cp_dim, l1000_dim = int(task_spec['target_dim']), int(task_spec['cp_dim']), int(task_spec['l1000_dim'])
        self.control_encoder = ControlEncoder(int(task_spec['context_dim']), hidden_dim)
        self.chemical_encoder = ChemicalEncoder(int(task_spec['chemical_dim']), hidden_dim)
        self.dose_encoder = DoseEncoder(int(task_spec['dose_dim']), hidden_dim)
        {fusion_setup}
        {bottleneck_setup}
        self.response = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout({dropout}), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        {readout_setup}
    def forward(self, pre, chemical, dose):
        ctrl, chem, dose_h = self.control_encoder(pre), self.chemical_encoder(chemical), self.dose_encoder(dose)
        {fusion_forward}
        {response_line}
        {readout_forward}

def build_model(task_spec): return CompiledMultimodalModel(task_spec)
def compute_loss(prediction, target, state):
    {loss_source}
def build_optimizer(model, state):
    del state
    return {optimizer_line}
def build_scheduler(optimizer, state):
    {scheduler_line}
'''


def compile_card(
    card: DesignCard,
    *,
    condition_dim: int,
    target_dim: int,
    base_choices: Mapping[str, object] | None = None,
) -> CompiledDesign:
    """Fail before training when metadata, edit scope, or tensor contract drift."""
    card.validate()
    instruction = instruction_for(card.slot)
    if instruction.mode != card.mode:
        raise DesignCardError("card mode does not match the frozen slot schedule")
    if condition_dim < 1 or target_dim < 2:
        raise DesignCardError("joint-output condition/target dimensions are invalid")
    choices = _choices(card, base_choices=base_choices)
    payload = {
        "schema_version": "cellscientist_source_constrained_compiled_design_v1",
        "card": asdict(card),
        "shape_contract": {
            "condition_dim": condition_dim,
            "target_dim": target_dim,
            "input_rank": 2,
            "output_rank": 2,
            "output_layout": "[n_samples, n_targets]",
        },
        "slot_instruction": asdict(instruction),
        "base_compiler_choices": dict(base_choices) if base_choices is not None else None,
        "compiler_choices": choices,
        "mechanism_certificate": _mechanism_certificate(fusion=str(choices["fusion"]), choices=choices),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return CompiledDesign(
        candidate_id=card.candidate_id,
        slot=card.slot,
        parent_candidate_id=card.parent_candidate_id,
        shape_contract=payload["shape_contract"],
        metadata=payload,
        digest=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def materialize_candidate_source(
    card: DesignCard,
    *,
    base_choices: Mapping[str, object] | None = None,
) -> tuple[str, Mapping[str, object]]:
    """Return compiler-owned source and metadata after all card checks."""
    card.validate()
    choices = _choices(card, base_choices=base_choices)
    source = _candidate_source(card=card, choices=choices)
    return source, _metadata(card=card, fusion=str(choices["fusion"]), readout=str(choices["readout"]), choices=choices)
