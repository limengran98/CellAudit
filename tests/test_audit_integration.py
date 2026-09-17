import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cellaudit.audit.endpoint_panel import (
    METHOD_ENDPOINT_FREEZE_FILE,
    OPEN_REFIT_COUNT,
    _aggregate_five_refit_status,
    _refit_open_source_for_fold5,
    _run_endpoint,
    _guided_checker_receipt,
    coordinate_utility_decomposition,
    freeze_panel,
    prepare_open_refits,
)
from cellaudit.cli import AUDIT_METHODS, _parser
from cellaudit.deterministic_claim_checker import IMPLEMENTED
from cellaudit.open_discovery import CandidateState
from cellaudit.joint_response_runtime import project_path


_SOURCE = """\
class Fusion:
    def forward(self, pre, chemical, dose):
        return pre + chemical + dose

def candidate_metadata():
    return {}

def build_model(task_spec):
    return Fusion()

def compute_loss(prediction, target, state):
    return prediction

def build_optimizer(model, state):
    return None
"""


def _candidate() -> CandidateState:
    return CandidateState(
        candidate_source=_SOURCE,
        candidate_metadata={
            "schema_version": "cellscientist_open_candidate_v1",
            "candidate_name": "frozen_open_endpoint",
            "semantic_components": ["perturbation.fusion"],
            "component_symbols": {"perturbation.fusion": ["Fusion"]},
            "change_summary": "fixture with explicit control, chemical, dose, and interaction paths",
            "claim_manifest": {
                "schema_version": "cellscientist_perturbation_claim_manifest_v1",
                "claims": [
                    {
                        "claim_type": "chemical_control_interaction",
                        "statement": "The fusion jointly consumes context and chemical identity.",
                        "source_symbols": ["Fusion"],
                    },
                    {
                        "claim_type": "dose_reliance",
                        "statement": "The fusion consumes the registered dose attribute.",
                        "source_symbols": ["Fusion"],
                    },
                ],
            },
        },
        training_config={},
    )


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class AuditIntegrationTests(unittest.TestCase):
    def test_open_refit_status_uses_four_of_five_agreement(self):
        self.assertEqual(
            _aggregate_five_refit_status(
                {"claim_supported": 4, "behaviorally_unsupported": 0, "inconclusive": 1, "not_identifiable": 0}
            ),
            "claim_supported",
        )
        self.assertEqual(
            _aggregate_five_refit_status(
                {"claim_supported": 0, "behaviorally_unsupported": 4, "inconclusive": 1, "not_identifiable": 0}
            ),
            "behaviorally_unsupported",
        )
        self.assertEqual(
            _aggregate_five_refit_status(
                {"claim_supported": 3, "behaviorally_unsupported": 2, "inconclusive": 0, "not_identifiable": 0}
            ),
            "inconclusive",
        )

    def test_coordinate_decomposition_closes_and_reports_interaction_separately(self):
        result = coordinate_utility_decomposition(
            correct_chemical_correct_dose=0.30,
            shuffled_chemical_correct_dose=0.26,
            correct_chemical_shuffled_dose=0.22,
            shuffled_chemical_shuffled_dose=0.20,
            context_anchor=0.24,
        )
        self.assertAlmostEqual(result["chemical_shapley"], 0.03)
        self.assertAlmostEqual(result["dose_shapley"], 0.07)
        self.assertAlmostEqual(result["both_shuffled_over_anchor"], -0.04)
        self.assertAlmostEqual(result["full_over_anchor"], 0.06)
        self.assertAlmostEqual(result["chemical_dose_interaction"], 0.02)
        self.assertAlmostEqual(result["closure_error"], 0.0)

    def test_prepare_open_refits_freezes_one_delivered_source_before_audit(self):
        runs = project_path("runs")
        runs.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="open_refit_prepare_", dir=runs) as temporary:
            base = Path(temporary)
            discovery = base / "open_campaign"
            refits = base / "open_refits"
            candidate = _candidate()
            result = discovery / "trajectory_01" / "result"
            candidate_root = result / "candidates" / "selected"
            _write_json(candidate_root / "candidate_state.json", candidate.to_dict())
            _write_json(
                discovery / "campaign_freeze.json",
                {
                    "schema_version": "cellscientist_discovery_campaign",
                    "status": "frozen_before_execution",
                    "method": "open",
                    "task_id": "cpg036_cp_plate_control_context",
                    "trajectory_count": 1,
                    "trajectory_seeds": [7001],
                },
            )
            _write_json(
                result / "summary.json",
                {
                    "schema_version": "fixture_open_summary",
                    "status": "complete",
                    "mode": "formal",
                    "task_id": "cpg036_cp_plate_control_context",
                    "fold4_or_fold5_used": False,
                    "trajectory_seed": 7001,
                    "selected_candidate_id": "selected",
                    "records": [
                        {
                            "candidate_id": "selected",
                            "status": "complete",
                            "feedback": {"global_pcc": 0.3},
                        }
                    ],
                },
            )
            (discovery / "_SUCCESS").write_text("complete\n", encoding="utf-8")

            def fake_refit(_model_id, **kwargs):
                checkpoint = kwargs["checkpoint_path"]
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(b"frozen-refit-checkpoint")
                return {
                    "status": "complete",
                    "selection_metrics": {
                        "global_pcc": 0.3,
                        "mse": 0.1,
                        "cp_pcc": 0.2,
                        "l1000_pcc": 0.4,
                    },
                    "checkpoint_path": str(checkpoint),
                }

            with patch("cellaudit.audit.endpoint_panel.load_fold13_arrays", return_value=object()), patch(
                "cellaudit.audit.endpoint_panel.train_joint_candidate", side_effect=fake_refit
            ):
                summary = prepare_open_refits(
                    task_id="cpg036_cp_plate_control_context",
                    discovery_root=discovery,
                    output_root=refits,
                    source_trajectory=1,
                    seed_base=7101,
                    device="cpu",
                )
            self.assertEqual(summary["refit_count"], OPEN_REFIT_COUNT)
            self.assertEqual(summary["provider_calls"], 0)
            hashes = set()
            for index in range(1, OPEN_REFIT_COUNT + 1):
                refit_result = refits / f"trajectory_{index:02d}" / "result"
                payload = json.loads((refit_result / "summary.json").read_text(encoding="utf-8"))
                state = json.loads(
                    (refit_result / "candidates" / "selected" / "candidate_state.json").read_text(encoding="utf-8")
                )
                hashes.add(state["candidate_hash"])
                self.assertEqual(payload["provider_calls"], 0)
                self.assertFalse(payload["fold4_or_fold5_used"])
            self.assertEqual(hashes, {candidate.candidate_hash})

    def test_prepare_open_refits_automatically_freezes_method_level_fold3_winner(self):
        runs = project_path("runs")
        runs.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="open_method_freeze_", dir=runs) as temporary:
            base = Path(temporary)
            discovery = base / "open_campaign"
            refits = base / "open_refits"
            candidate = _candidate()
            _write_json(
                discovery / "campaign_freeze.json",
                {
                    "schema_version": "cellscientist_discovery_campaign",
                    "status": "frozen_before_execution",
                    "method": "open",
                    "task_id": "cpg036_cp_plate_control_context",
                    "trajectory_count": 10,
                    "trajectory_seeds": [7000 + index for index in range(1, 11)],
                },
            )
            for trajectory in range(1, 11):
                candidate_id = f"candidate_{trajectory:02d}"
                pcc, mse, parameters = 0.20 + trajectory / 1000.0, 0.10, 100
                if trajectory == 2:
                    pcc, mse, parameters = 0.50, 0.20, 50
                elif trajectory == 3:
                    pcc, mse, parameters = 0.50, 0.10, 100
                elif trajectory == 4:
                    pcc, mse, parameters = 0.50, 0.10, 200
                elif trajectory == 5:
                    pcc, mse, parameters = 0.50, 0.10, 100
                result = discovery / f"trajectory_{trajectory:02d}" / "result"
                state = result / "candidates" / candidate_id / "candidate_state.json"
                _write_json(state, candidate.to_dict())
                record = {
                    "candidate_id": candidate_id,
                    "candidate_hash": candidate.candidate_hash,
                    "status": "complete",
                    "feedback": {
                        "fold3_global_pcc": pcc,
                        "fold3_mse": mse,
                        "parameter_count": parameters,
                    },
                }
                _write_json(
                    result / "summary.json",
                    {
                        "schema_version": "fixture_open_summary",
                        "status": "complete",
                        "mode": "formal",
                        "task_id": "cpg036_cp_plate_control_context",
                        "fold4_or_fold5_used": False,
                        "trajectory_seed": 7000 + trajectory,
                        "selected_candidate_id": candidate_id,
                        "records": [record],
                    },
                )
            (discovery / "_SUCCESS").write_text("complete\n", encoding="utf-8")

            def fake_refit(_model_id, **kwargs):
                checkpoint = kwargs["checkpoint_path"]
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(b"method-endpoint-refit")
                return {
                    "status": "complete",
                    "selection_metrics": {
                        "global_pcc": 0.5,
                        "mse": 0.1,
                        "cp_pcc": 0.4,
                        "l1000_pcc": 0.6,
                    },
                    "checkpoint_path": str(checkpoint),
                }

            with patch("cellaudit.audit.endpoint_panel.load_fold13_arrays", return_value=object()), patch(
                "cellaudit.audit.endpoint_panel.train_joint_candidate", side_effect=fake_refit
            ):
                summary = prepare_open_refits(
                    task_id="cpg036_cp_plate_control_context",
                    discovery_root=discovery,
                    output_root=refits,
                    seed_base=7101,
                    device="cpu",
                )
            endpoint_freeze = json.loads(
                (refits / METHOD_ENDPOINT_FREEZE_FILE).read_text(encoding="utf-8")
            )
            self.assertEqual(endpoint_freeze["executable_candidate_count"], 10)
            self.assertEqual(endpoint_freeze["selected"]["trajectory"], 3)
            self.assertEqual(endpoint_freeze["selected"]["candidate_id"], "candidate_03")
            self.assertEqual(summary["source_trajectory"], 3)
            self.assertEqual(summary["source_candidate_id"], "candidate_03")

    def test_fold5_open_refit_reuses_source_and_seed_not_fold4_weights(self):
        runs = project_path("runs")
        runs.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="open_fold5_refit_", dir=runs) as temporary:
            base = Path(temporary)
            candidate = _candidate()
            state_path = base / "candidate_state.json"
            _write_json(state_path, candidate.to_dict())
            old_checkpoint = base / "fold4_checkpoint.pt"
            old_checkpoint.write_bytes(b"fold4-weights")
            unit_root = base / "fold5_unit"
            unit_root.mkdir()
            record = {
                "candidate_hash": candidate.candidate_hash,
                "candidate_state": {
                    "path": str(state_path.relative_to(project_path("."))),
                    "sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
                },
                "checkpoint": {
                    "path": str(old_checkpoint.relative_to(project_path("."))),
                    "sha256": hashlib.sha256(old_checkpoint.read_bytes()).hexdigest(),
                },
                "trajectory_seed": 7201,
            }
            observed = {}

            def fake_refit(_model_id, **kwargs):
                observed["seed"] = kwargs["settings"].seed
                observed["arrays"] = kwargs["arrays"]
                checkpoint = kwargs["checkpoint_path"]
                checkpoint.write_bytes(b"new-fold5-weights")
                return {"status": "complete", "selection_metrics": {"global_pcc": 0.3}}

            fold13 = object()
            with patch("cellaudit.audit.endpoint_panel.load_fold13_arrays", return_value=fold13), patch(
                "cellaudit.audit.endpoint_panel.train_joint_candidate", side_effect=fake_refit
            ):
                replay, receipt = _refit_open_source_for_fold5(
                    record=record,
                    task_id="cpg036_cp_plate_control_context",
                    unit_root=unit_root,
                    device="cpu",
                )
            self.assertEqual(observed, {"seed": 7201, "arrays": fold13})
            self.assertNotEqual(replay["checkpoint"]["sha256"], record["checkpoint"]["sha256"])
            self.assertEqual(receipt["candidate_hash"], candidate.candidate_hash)
            self.assertFalse(receipt["fold4_or_fold5_used_for_fitting_or_selection"])

    def test_fold5_response_is_not_loaded_before_open_source_refit(self):
        runs = project_path("runs")
        runs.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="fold5_order_", dir=runs) as temporary:
            root = Path(temporary)
            record = {"audit_unit_id": "open_refit_01"}
            with patch(
                "cellaudit.audit.endpoint_panel._load_frozen",
                return_value=({}, {"method": "open", "task_id": "cpg036_cp_plate_control_context"}),
            ), patch(
                "cellaudit.audit.endpoint_panel._refit_open_source_for_fold5",
                side_effect=RuntimeError("stop after refit entry"),
            ), patch("cellaudit.audit.endpoint_panel.load_fold5_arrays") as fold5_loader:
                with self.assertRaisesRegex(RuntimeError, "stop after refit entry"):
                    _run_endpoint(root=root, record=record, fold=5, device="cpu")
            fold5_loader.assert_not_called()

    def test_open_five_refit_freeze_runs_checker_for_every_registered_claim(self):
        runs = project_path("runs")
        runs.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="audit_contract_", dir=runs) as temporary:
            base = Path(temporary)
            discovery = base / "open_refits"
            output = base / "audit"
            candidate = _candidate()
            seeds = [7101 + index for index in range(OPEN_REFIT_COUNT)]
            method_freeze = discovery / METHOD_ENDPOINT_FREEZE_FILE
            _write_json(
                method_freeze,
                {
                    "schema_version": "cellscientist_open_method_endpoint_freeze_v1",
                    "status": "frozen_before_refitting",
                    "method": "open",
                    "task_id": "cpg036_cp_plate_control_context",
                    "selected": {"candidate_hash": candidate.candidate_hash},
                },
            )
            _write_json(
                discovery / "campaign_freeze.json",
                {
                    "schema_version": "cellscientist_discovery_campaign",
                    "status": "frozen_before_execution",
                    "method": "open",
                    "task_id": "cpg036_cp_plate_control_context",
                    "trajectory_count": OPEN_REFIT_COUNT,
                    "trajectory_seeds": seeds,
                    "candidate_hash": candidate.candidate_hash,
                    "method_endpoint_freeze": {
                        "path": str(method_freeze.resolve().relative_to(project_path("."))),
                        "sha256": hashlib.sha256(method_freeze.read_bytes()).hexdigest(),
                    },
                },
            )
            (discovery / "_SUCCESS").write_text("complete\n", encoding="utf-8")
            for index, seed in enumerate(seeds, start=1):
                result = discovery / f"trajectory_{index:02d}" / "result"
                candidate_root = result / "candidates" / "selected"
                _write_json(candidate_root / "candidate_state.json", candidate.to_dict())
                (candidate_root / "selected_checkpoint.pt").write_bytes(f"checkpoint-{seed}".encode())
                _write_json(
                    result / "summary.json",
                    {
                        "schema_version": "fixture_open_refit_summary",
                        "status": "complete",
                        "mode": "formal",
                        "task_id": "cpg036_cp_plate_control_context",
                        "fold4_or_fold5_used": False,
                        "trajectory_seed": seed,
                        "selected_candidate_id": "selected",
                        "records": [
                            {
                                "candidate_id": "selected",
                                "status": "complete",
                                "feedback": {"global_pcc": 0.3},
                            }
                        ],
                    },
                )

            freeze_panel(
                method="open",
                task_id="cpg036_cp_plate_control_context",
                discovery_root=discovery,
                output_root=output,
                trajectory_count=OPEN_REFIT_COUNT,
            )
            fold4_freeze = json.loads((output / "FOLD4_AUDIT_FREEZE.json").read_text(encoding="utf-8"))
            self.assertIn("runtime_config", fold4_freeze)
            self.assertIn("method_endpoint_freeze", fold4_freeze)
            registry = json.loads((output / "ALL_ENDPOINT_REGISTRY.json").read_text(encoding="utf-8"))
            self.assertEqual(registry["method"], "open")
            self.assertEqual(registry["panel_design"], "five_paired_refits_of_one_frozen_source")
            self.assertEqual(len(registry["records"]), OPEN_REFIT_COUNT)
            self.assertEqual(len({row["candidate_hash"] for row in registry["records"]}), 1)
            for row in registry["records"]:
                receipt_path = project_path(row["checker_receipt"]["path"])
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                self.assertEqual(len(receipt["claims"]), 4)
                self.assertEqual(
                    {claim["static_status"] for claim in receipt["claims"]},
                    {IMPLEMENTED},
                )
                self.assertEqual(row["source_contract"]["checker_receipt_hash"], receipt["receipt_hash"])

    def test_cli_registers_open_audit_method(self):
        self.assertEqual(AUDIT_METHODS["open"], "open")
        args = _parser().parse_args(
            [
                "audit",
                "freeze",
                "--method",
                "open",
                "--task",
                "bbbc036",
                "--discovery-root",
                "runs/refits",
                "--output-root",
                "runs/audit",
            ]
        )
        self.assertEqual(args.method, "open")
        prepare = _parser().parse_args(
            [
                "audit",
                "prepare-open-refits",
                "--task",
                "bbbc036",
                "--discovery-root",
                "runs/open",
                "--output-root",
                "runs/refits",
            ]
        )
        self.assertEqual(prepare.action, "prepare-open-refits")

    def test_falsification_guided_source_uses_the_same_checker_taxonomy(self):
        receipt, candidate_hash = _guided_checker_receipt(
            candidate_id="guided_endpoint",
            design={
                "fusion": "factorized_gate",
                "hidden_dim": 384,
                "readout": "dual_heads",
                "objective": "balanced_residual",
            },
            fold3_global_pcc=0.31,
        )
        self.assertEqual(receipt["candidate_hash"], candidate_hash)
        self.assertEqual(len(receipt["claims"]), 4)
        self.assertEqual(
            {claim["static_status"] for claim in receipt["claims"]},
            {IMPLEMENTED},
        )


if __name__ == "__main__":
    unittest.main()
