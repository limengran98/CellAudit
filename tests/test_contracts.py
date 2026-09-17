from pathlib import Path
import importlib
import sys
import unittest

from cellaudit.cli import TASKS
from cellaudit.candidate_materialization import (
    ExecutablePatchManifestProposal,
    PatchEditProposal,
    TopLevelSymbolOperation,
    normalize_patch_manifest,
)
from cellaudit.discovery.falsification_guided.model import FalsificationGuidedDesign
from cellaudit.provider import provider_lock_for_panel_key, validate_provider_lock
from cellaudit.tasks import get_task
from cellaudit.joint_response_runtime import load_runtime_config, load_starting_candidate


class PublicContractTests(unittest.TestCase):
    def test_package_namespace_and_entry_points(self):
        root = Path(__file__).resolve().parents[1]
        package = importlib.import_module("cellaudit")
        self.assertEqual(Path(package.__file__).resolve().parent, root / "cellaudit")
        self.assertFalse((root / "cellscientist").exists())
        project = (root / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('cellaudit = "cellaudit.cli:main"', project)
        self.assertIn('include = ["cellaudit*"]', project)
        for script in (root / "scripts").glob("*.sh"):
            text = script.read_text(encoding="utf-8")
            self.assertIn("python -m cellaudit", text)
            self.assertNotIn("python -m " + "cellscientist", text)

    def test_registered_tasks_and_fold_roles(self):
        runtime = load_runtime_config("configs/joint_response_runtime.json")
        self.assertEqual(
            runtime["fold_roles"],
            {"fit_folds": [1, 2], "selection_fold": 3, "withheld_folds": [4, 5]},
        )
        self.assertEqual(set(TASKS), {"bbbc036", "bbbc047"})
        for task_id in TASKS.values():
            task = get_task(task_id)
            self.assertEqual(task["condition_layout"]["prefix_dim"], 2048)

    def test_frozen_candidate_receipts_replay(self):
        runtime = load_runtime_config("configs/joint_response_runtime.json")
        candidate = load_starting_candidate("h0", runtime)
        self.assertEqual(len(candidate.candidate_hash), 64)
        self.assertIn("build_model", candidate.candidate_source)

    def test_provider_lock_has_no_fallback(self):
        lock = provider_lock_for_panel_key("deepseek_v4_flash")
        self.assertEqual(validate_provider_lock(lock).model_id, "deepseek-v4-flash")
        self.assertFalse(lock["allow_fallback"])

    def test_falsification_guided_design_contract(self):
        design = FalsificationGuidedDesign(
            fusion="centered_additive",
            hidden_dim=256,
            readout="shared",
            objective="balanced_residual",
        )
        design.validate()
        self.assertEqual(design.to_dict()["fusion"], "centered_additive")

    def test_public_tree_has_no_machine_paths(self):
        root = Path(__file__).resolve().parents[1]
        text = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in root.rglob("*")
            if path.is_file() and path.suffix in {".py", ".json", ".md", ".toml", ".sh"}
        )
        self.assertNotIn("/private/" + "workspace", text)
        self.assertNotIn("next_" + "versions", text)
        self.assertNotIn("cellscientist_" + "cdr", text)
        self.assertNotIn("Luo" + "SS", text)
        self.assertNotIn(" C" + "CI ", text)

    def test_public_tree_contains_method_code_not_result_tables(self):
        root = Path(__file__).resolve().parents[1]
        self.assertFalse((root / "evidence").exists())
        self.assertTrue((root / "cellaudit" / "deterministic_claim_checker.py").is_file())
        self.assertTrue((root / "cellaudit" / "audit" / "endpoint_panel.py").is_file())

    def test_open_discovery_import_chain_has_no_legacy_modules(self):
        root = Path(__file__).resolve().parents[1]
        package = root / "cellaudit"
        old_contract_module = "revision_" + "gr" + "aph"
        old_materializer_module = "revision_" + "materializer"
        self.assertFalse((package / f"{old_contract_module}.py").exists())
        self.assertFalse((package / f"{old_materializer_module}.py").exists())
        self.assertTrue((package / "candidate_change_contracts.py").is_file())
        self.assertTrue((package / "candidate_materialization.py").is_file())

        importlib.import_module("cellaudit.open_discovery")
        imported = set(sys.modules)
        self.assertNotIn(f"cellaudit.{old_contract_module}", imported)
        self.assertNotIn(f"cellaudit.{old_materializer_module}", imported)

        public_entry_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                package / "open_discovery.py",
                package / "cli.py",
                root / "configs" / "open_discovery.json",
            )
        ).lower()
        for forbidden in (
            old_contract_module,
            old_materializer_module,
            "executable-" + "gr" + "aph",
            "uses_" + "gr" + "aph_or_" + "br" + "idge",
            "rou" + "ter-visible",
        ):
            self.assertNotIn(forbidden, public_entry_text)

    def test_candidate_change_materialization_is_self_contained(self):
        parent = "def build_model():\n    return None\n"
        child = parent + "\ndef response_helper(x):\n    return x\n"
        proposal = ExecutablePatchManifestProposal(
            manifest_id="manifest_smoke",
            edits=(
                PatchEditProposal(
                    edit_id="edit_response_helper",
                    address="response.helper",
                    kind="add_helper",
                    description="Add one response helper.",
                    dependencies=(),
                    incompatibilities=(),
                    preconditions=(),
                    symbol_operations=(
                        TopLevelSymbolOperation(
                            "add",
                            "response_helper",
                            "def response_helper(x):\n    return x\n",
                        ),
                    ),
                    config_merge_patch={},
                ),
            ),
        )
        normalized = normalize_patch_manifest(
            proposal,
            parent_source=parent,
            observed_child_source=child,
            infer_support_dependencies=False,
        )
        self.assertEqual(normalized.edits[0].symbol_operations[0].operation, "add")
        self.assertEqual(
            normalized.edits[0].symbol_operations[0].symbol_name,
            "response_helper",
        )

    def test_release_has_no_legacy_orchestration_vocabulary(self):
        root = Path(__file__).resolve().parents[1]
        text = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore").lower()
            for path in root.rglob("*")
            if path.is_file()
            and path.name != "MANIFEST.sha256"
            and path.suffix in {".py", ".json", ".md", ".toml", ".sh"}
        )
        forbidden = (
            "rou" + "ter",
            "rou" + "ting",
            "route" + "able",
            "br" + "idge",
            "revision_" + "gr" + "aph",
            "revision " + "gr" + "aph",
            "executable-" + "gr" + "aph",
        )
        for term in forbidden:
            self.assertNotIn(term, text)

    def test_release_contains_only_method_runtime(self):
        root = Path(__file__).resolve().parents[1]
        text = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore").lower()
            for path in root.rglob("*")
            if path.is_file()
            and path.name != "MANIFEST.sha256"
            and path.suffix in {".py", ".json", ".md", ".toml", ".sh"}
        )
        forbidden = (
            "ai" + "de",
            "cell" + "forge",
            "harmony" + "cell",
            "real" + "mlp",
            "tab" + "m",
            "tab" + "r",
            "ri" + "dge",
            "standard_" + "mlp",
            "standard " + "mlp",
            "unified_cpg_" + "baselines",
            "configs/" + "baselines.json",
            "assets/" + "baselines",
        )
        for term in forbidden:
            self.assertNotIn(term, text)
        self.assertFalse((root / "cellaudit" / ("com" + "piler.py")).exists())
        self.assertTrue((root / "cellaudit" / "joint_response_runtime.py").is_file())


if __name__ == "__main__":
    unittest.main()
