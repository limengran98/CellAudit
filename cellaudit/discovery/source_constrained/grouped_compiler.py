"""Deterministic compiler for source-constrained discovery's grouped multimodal candidate language."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping

from .types import DesignCard, DesignCardError
from .grouped_schedule import instruction_for


@dataclass(frozen=True)
class CompiledDesign:
    candidate_id: str
    slot: int
    parent_candidate_id: str | None
    shape_contract: Mapping[str, int | str]
    metadata: Mapping[str, object]
    digest: str


FUSIONS = frozenset({
    "film", "gated_residual", "multitoken_attention", "bilinear_gate",
    "context_residual_adapter",
})
READOUTS = frozenset({"shared", "dual_heads", "group_experts"})
HIDDEN = frozenset({256, 384, 512})
LOSSES = frozenset({"mse", "pcc_mse", "group_balanced_mse", "group_balanced_pcc_mse"})


def default_choices() -> dict[str, object]:
    return {
        "fusion": "context_residual_adapter",
        "readout": "shared",
        "hidden_dim": 256,
        "dropout": 0.1,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "loss": "mse",
        "bottleneck": False,
        "deep_chemical_encoder": False,
        "residual_response": False,
    }


def choices_for(card: DesignCard, *, base_choices: Mapping[str, object] | None = None) -> dict[str, object]:
    defaults = default_choices()
    if base_choices is None:
        values = defaults
    else:
        if set(base_choices) != set(defaults):
            raise DesignCardError("parent compiler choices do not match the source-constrained discovery grouped candidate language")
        values = dict(base_choices)
    for edit in card.edits:
        token = edit.mechanism.strip().lower().replace("-", "_")
        if edit.component == "fusion":
            if token not in FUSIONS:
                raise DesignCardError("fusion mechanism is not compiler-registered")
            values["fusion"] = token
        elif edit.component == "response_readout":
            if token not in READOUTS:
                raise DesignCardError("response_readout must be shared, dual_heads, or group_experts")
            values["readout"] = token
        elif edit.component == "model_architecture":
            if not token.startswith("hidden_") or not token.removeprefix("hidden_").isdigit():
                raise DesignCardError("model_architecture must declare a registered hidden width")
            width = int(token.removeprefix("hidden_"))
            if width not in HIDDEN:
                raise DesignCardError("hidden width is not compiler-registered")
            values["hidden_dim"] = width
        elif edit.component == "chemical_encoder":
            if token == "standard":
                values["deep_chemical_encoder"] = False
            elif token == "deep_two_layer":
                values["deep_chemical_encoder"] = True
            else:
                raise DesignCardError("chemical_encoder must be standard or deep_two_layer")
        elif edit.component == "response_program":
            if token == "standard":
                values["bottleneck"] = False
                values["residual_response"] = False
            elif token == "bottleneck_64":
                values["bottleneck"] = True
                values["residual_response"] = False
            elif token == "residual":
                values["bottleneck"] = False
                values["residual_response"] = True
            else:
                raise DesignCardError("response_program must be standard, bottleneck_64, or residual")
        elif edit.component == "loss":
            if token not in LOSSES:
                raise DesignCardError("loss mechanism is not compiler-registered")
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


def _mechanism_certificate(choices: Mapping[str, object]) -> dict[str, object]:
    fusion = str(choices["fusion"])
    paths = {
        "film": "condition modulation of the context representation",
        "gated_residual": "condition residual gated against the context representation",
        "bilinear_gate": "context-conditioned gate applied to the perturbation representation",
        "multitoken_attention": "three-token context/perturbation/dose attention",
        "context_residual_adapter": "context-dose anchor plus a nonzero gated perturbation adapter",
    }
    return {
        "schema_version": "cellscientist_source_constrained_mechanism_certificate_v2",
        "declared_input_contract": "perturbation_context_conditioned_response",
        "perturbation_path": paths[fusion],
        "single_token_attention_forbidden": True,
        "attention_token_count": 3 if fusion == "multitoken_attention" else 0,
        "grouped_output_program": str(choices["readout"]),
        "endpoint_heldout_behavioral_audit_required": True,
    }


def _metadata(card: DesignCard, choices: Mapping[str, object]) -> dict[str, object]:
    fusion, readout = str(choices["fusion"]), str(choices["readout"])
    return {
        "schema_version": "cellscientist_open_candidate_v1",
        "candidate_name": f"candidate_{card.candidate_id}_{fusion}_{readout}",
        "semantic_components": [
            "input.context_encoding", "input.perturbation_encoding", "input.dose_encoding",
            f"perturbation.{fusion}", "response.shared_program", f"response.{readout}",
            "objective.grouped_response_loss", "optimization.optimizer", "optimization.scheduler",
        ],
        "component_symbols": {
            "input.context_encoding": ["ContextEncoder"],
            "input.perturbation_encoding": ["PerturbationEncoder"],
            "input.dose_encoding": ["DoseEncoder"],
            f"perturbation.{fusion}": ["CompiledGroupedResponseModel"],
            "response.shared_program": ["CompiledGroupedResponseModel"],
            f"response.{readout}": ["CompiledGroupedResponseModel"],
            "objective.grouped_response_loss": ["compute_loss"],
            "optimization.optimizer": ["build_optimizer"],
            "optimization.scheduler": ["build_scheduler"],
        },
        "change_summary": f"Deterministic materialization of design card {card.candidate_id}: {card.hypothesis}",
        "mechanism_certificate": _mechanism_certificate(choices),
    }


def _candidate_source(card: DesignCard, choices: Mapping[str, object]) -> str:
    fusion = str(choices["fusion"])
    readout = str(choices["readout"])
    hidden = int(choices["hidden_dim"])
    dropout = float(choices["dropout"])
    optimizer = str(choices["optimizer"])
    scheduler = str(choices["scheduler"])
    loss_name = str(choices["loss"])
    bottleneck = bool(choices["bottleneck"])
    deep_chemical = bool(choices["deep_chemical_encoder"])
    residual_response = bool(choices["residual_response"])

    fusion_setup = {
        "film": "self.modulator = nn.Linear(hidden_dim * 2, hidden_dim * 2)",
        "gated_residual": "self.gate = nn.Linear(hidden_dim * 2, hidden_dim)",
        "bilinear_gate": "self.perturbation_gate = nn.Linear(hidden_dim, hidden_dim)",
        "multitoken_attention": "self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)",
        "context_residual_adapter": "self.context_value = nn.Linear(hidden_dim, hidden_dim); self.dose_value = nn.Linear(hidden_dim, hidden_dim); self.perturbation_value = nn.Linear(hidden_dim, hidden_dim); self.adapter_gate = nn.Linear(hidden_dim * 3, hidden_dim); self.adapter_scale = nn.Parameter(torch.tensor(-2.0)); self.fusion_norm = nn.LayerNorm(hidden_dim)",
    }[fusion]
    fusion_forward = {
        "film": "params = self.modulator(torch.cat((pert, dose_h), dim=1)); scale, bias = params.chunk(2, dim=1); fused = ctrl * (1.0 + torch.tanh(scale)) + bias",
        "gated_residual": "gate = torch.sigmoid(self.gate(torch.cat((pert, dose_h), dim=1))); fused = ctrl + gate * (pert + dose_h - ctrl)",
        "bilinear_gate": "fused = ctrl + dose_h + pert * torch.sigmoid(self.perturbation_gate(ctrl))",
        "multitoken_attention": "tokens = torch.stack((ctrl, pert, dose_h), dim=1); attended, _ = self.attention(tokens, tokens, tokens, need_weights=False); fused = attended[:, 0, :] + tokens[:, 0, :]",
        "context_residual_adapter": "base = self.context_value(ctrl) + self.dose_value(dose_h); gate = torch.sigmoid(self.adapter_gate(torch.cat((ctrl, pert, dose_h), dim=1))); delta = gate * self.perturbation_value(pert); fused = self.fusion_norm(base + torch.sigmoid(self.adapter_scale) * delta)",
    }[fusion]
    chemical_setup = (
        "self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())"
        if deep_chemical else
        "self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())"
    )
    bottleneck_setup = (
        "self.bottleneck = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, hidden_dim))"
        if bottleneck else "self.bottleneck = nn.Identity()"
    )
    if readout == "shared":
        readout_setup = "self.readout = nn.Linear(hidden_dim, target_dim)"
        readout_forward = "return self.readout(hidden)"
    elif readout == "dual_heads":
        readout_setup = "self.group1_readout = nn.Linear(hidden_dim, group1_dim); self.group2_readout = nn.Linear(hidden_dim, group2_dim)"
        readout_forward = "return torch.cat((self.group1_readout(hidden), self.group2_readout(hidden)), dim=1)"
    else:
        readout_setup = "self.group1_expert = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)); self.group2_expert = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)); self.group1_norm = nn.LayerNorm(hidden_dim); self.group2_norm = nn.LayerNorm(hidden_dim); self.group1_readout = nn.Linear(hidden_dim, group1_dim); self.group2_readout = nn.Linear(hidden_dim, group2_dim)"
        readout_forward = "group1_hidden = self.group1_norm(hidden + self.group1_expert(hidden)); group2_hidden = self.group2_norm(hidden + self.group2_expert(hidden)); return torch.cat((self.group1_readout(group1_hidden), self.group2_readout(group2_hidden)), dim=1)"
    response_line = (
        "hidden = self.response(self.bottleneck(fused)); hidden = hidden + fused"
        if residual_response else "hidden = self.response(self.bottleneck(fused))"
    )
    loss_source = {
        "mse": "return torch.nn.functional.mse_loss(prediction, target)",
        "pcc_mse": "mse = torch.nn.functional.mse_loss(prediction, target); pred = prediction - prediction.mean(dim=1, keepdim=True); truth = target - target.mean(dim=1, keepdim=True); corr = (pred * truth).sum(dim=1) / (pred.square().sum(dim=1).sqrt() * truth.square().sum(dim=1).sqrt() + 1e-8); return mse + 0.1 * (1.0 - corr.mean())",
        # The protected interface validator intentionally supplies no task
        # metadata to candidate loss code.  It uses a shape-only half split;
        # physical training supplies the exact registered group boundary.
        "group_balanced_mse": "group1_dim = int(state.get('cp_dim', prediction.shape[1] // 2)); group1 = torch.nn.functional.mse_loss(prediction[:, :group1_dim], target[:, :group1_dim]); group2 = torch.nn.functional.mse_loss(prediction[:, group1_dim:], target[:, group1_dim:]); return 0.5 * (group1 + group2)",
        "group_balanced_pcc_mse": "group1_dim = int(state.get('cp_dim', prediction.shape[1] // 2)); losses = [];\n    for pred_group, true_group in ((prediction[:, :group1_dim], target[:, :group1_dim]), (prediction[:, group1_dim:], target[:, group1_dim:])):\n        mse = torch.nn.functional.mse_loss(pred_group, true_group); pred_centered = pred_group - pred_group.mean(dim=1, keepdim=True); true_centered = true_group - true_group.mean(dim=1, keepdim=True); corr = (pred_centered * true_centered).sum(dim=1) / (pred_centered.square().sum(dim=1).sqrt() * true_centered.square().sum(dim=1).sqrt() + 1e-8); losses.append(mse + 0.05 * (1.0 - corr.mean()))\n    return 0.5 * (losses[0] + losses[1])",
    }[loss_name]
    optimizer_line = (
        "torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)"
        if optimizer == "adamw" else "torch.optim.Adam(model.parameters(), lr=1e-3)"
    )
    scheduler_line = (
        "return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(state['max_epochs']))"
        if scheduler == "cosine" else "return None"
    )
    metadata = _metadata(card, choices)
    return f'''import torch
from torch import nn

def candidate_metadata():
    return {metadata!r}

class ContextEncoder(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__(); self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
    def forward(self, value): return self.net(value)

class PerturbationEncoder(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__(); {chemical_setup}
    def forward(self, value): return self.net(value)

class DoseEncoder(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__(); self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
    def forward(self, value): return self.net(value)

class CompiledGroupedResponseModel(nn.Module):
    def __init__(self, task_spec):
        super().__init__()
        hidden_dim, dropout = {hidden}, {dropout}
        target_dim = int(task_spec['target_dim'])
        group1_dim, group2_dim = int(task_spec['cp_dim']), int(task_spec['l1000_dim'])
        self.context_encoder = ContextEncoder(int(task_spec['context_dim']), hidden_dim)
        self.perturbation_encoder = PerturbationEncoder(int(task_spec['chemical_dim']), hidden_dim)
        self.dose_encoder = DoseEncoder(int(task_spec['dose_dim']), hidden_dim)
        {fusion_setup}
        {bottleneck_setup}
        self.response = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        {readout_setup}
    def forward(self, pre, chemical, dose):
        ctrl, pert, dose_h = self.context_encoder(pre), self.perturbation_encoder(chemical), self.dose_encoder(dose)
        {fusion_forward}
        {response_line}
        {readout_forward}

def build_model(task_spec): return CompiledGroupedResponseModel(task_spec)
def compute_loss(prediction, target, state):
    {loss_source}
def build_optimizer(model, state):
    del state
    return {optimizer_line}
def build_scheduler(optimizer, state):
    {scheduler_line}
'''


def compile_card(card: DesignCard, *, condition_dim: int, target_dim: int,
                 base_choices: Mapping[str, object] | None = None) -> CompiledDesign:
    card.validate()
    instruction = instruction_for(card.slot)
    if instruction.mode != card.mode:
        raise DesignCardError("card mode does not match the grouped multimodal schedule")
    if len(card.edits) > instruction.max_component_edits:
        raise DesignCardError("card exceeds the registered slot edit budget")
    if instruction.exact_edits:
        actual_edits = tuple((edit.component, edit.mechanism.strip().lower().replace("-", "_")) for edit in card.edits)
        if actual_edits != instruction.exact_edits:
            raise DesignCardError("card must implement the slot's exact orthogonal design edit")
    if instruction.required_components and not any(
        edit.component in instruction.required_components for edit in card.edits
    ):
        raise DesignCardError("card does not address the registered diagnostic component for this slot")
    if card.slot <= 3 and sum(edit.component == "fusion" for edit in card.edits) != 1:
        raise DesignCardError("complete anchor hypotheses require exactly one explicit fusion")
    if condition_dim < 1 or target_dim < 2:
        raise DesignCardError("joint-output condition/target dimensions are invalid")
    choices = choices_for(card, base_choices=base_choices)
    payload = {
        "schema_version": "cellscientist_source_constrained_grouped_compiled_design_v1",
        "card": asdict(card),
        "shape_contract": {
            "condition_dim": condition_dim, "target_dim": target_dim,
            "input_rank": 2, "output_rank": 2,
            "output_layout": "[n_samples, n_targets]",
        },
        "slot_instruction": asdict(instruction),
        "base_compiler_choices": dict(base_choices) if base_choices is not None else None,
        "compiler_choices": choices,
        "mechanism_certificate": _mechanism_certificate(choices),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return CompiledDesign(
        candidate_id=card.candidate_id, slot=card.slot,
        parent_candidate_id=card.parent_candidate_id,
        shape_contract=payload["shape_contract"], metadata=payload,
        digest=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def materialize_candidate_source(card: DesignCard, *, base_choices: Mapping[str, object] | None = None) -> tuple[str, Mapping[str, object]]:
    choices = choices_for(card, base_choices=base_choices)
    return _candidate_source(card, choices), _metadata(card, choices)
