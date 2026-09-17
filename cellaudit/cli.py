"""Command-line entry points for discovery and deterministic audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .audit.endpoint_panel import (
    CONSTRAINED_ENDPOINT_COUNT,
    OPEN_REFIT_COUNT,
    freeze_fold5,
    freeze_panel,
    prepare_open_refits,
    run_all,
    summarize,
)
from .bounded_discovery import run_bounded_discovery
from .discovery.falsification_guided.runtime import run_falsification_guided_discovery
from .discovery.source_constrained.runtime import run_source_constrained_discovery
from .schemas import canonical_json
from .tasks import get_task
from .joint_response_runtime import load_runtime_config, load_starting_candidate, project_path


TASKS = {
    "bbbc036": "cpg036_cp_plate_control_context",
    "bbbc047": "cpg047_cp_plate_control_context",
}

STAGES: Mapping[str, tuple[Callable[..., Mapping[str, Any]], str]] = {
    "open": (run_bounded_discovery, "configs/open_discovery.json"),
    "source-constrained": (
        run_source_constrained_discovery,
        "configs/source_constrained_discovery.json",
    ),
    "falsification-guided": (
        run_falsification_guided_discovery,
        "configs/falsification_guided_discovery.json",
    ),
}

AUDIT_METHODS = {
    "open": "open",
    "source-constrained": "source_constrained",
    "falsification-guided": "falsification_guided",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _task_id(value: str) -> str:
    return TASKS[value]


def _discover(args: argparse.Namespace) -> Mapping[str, Any]:
    runner, config = STAGES[args.stage]
    kwargs: dict[str, Any] = {
        "task_id": _task_id(args.task),
        "output_root": args.output_root,
        "mode": args.mode,
        "device": args.device,
        "config_path": config,
        "trajectory_seed": args.seed,
    }
    if args.stage != "open" and args.slots is not None:
        kwargs["proposal_slots"] = args.slots
    return runner(**kwargs)


def _campaign(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.trajectories != 10:
        raise ValueError("the formal review protocol registers exactly ten trajectories")
    runner, config = STAGES[args.stage]
    root = project_path(args.output_root)
    if root.exists() or not root.is_relative_to(project_path("runs")):
        raise ValueError("campaign output must be a new directory below runs/")
    root.mkdir(parents=True, exist_ok=False)
    seeds = [int(args.seed_base) + index for index in range(args.trajectories)]
    method = AUDIT_METHODS.get(args.stage, "open")
    config_file = project_path(config)
    freeze = {
        "schema_version": "cellscientist_discovery_campaign",
        "status": "frozen_before_execution",
        "method": method,
        "task_id": _task_id(args.task),
        "trajectory_count": args.trajectories,
        "trajectory_seeds": seeds,
        "candidate_budget": 10,
        "config": str(config_file.relative_to(project_path("."))),
        "config_sha256": _sha256(config_file),
        "fold_roles": {"fit": [1, 2], "selection": 3, "withheld": [4, 5]},
    }
    _write(root / "campaign_freeze.json", freeze)
    completed: list[Mapping[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        result_root = root / f"trajectory_{index:02d}" / "result"
        kwargs: dict[str, Any] = {
            "task_id": _task_id(args.task),
            "output_root": result_root,
            "mode": "formal",
            "device": args.device,
            "config_path": config,
            "trajectory_seed": seed,
        }
        result = runner(**kwargs)
        completed.append({
            "trajectory": index,
            "seed": seed,
            "status": result.get("status"),
            "selected_candidate_id": result.get("selected_candidate_id"),
        })
    summary = {
        "schema_version": "cellscientist_discovery_campaign_summary",
        "status": "complete",
        "method": method,
        "task_id": _task_id(args.task),
        "trajectory_count": len(completed),
        "trajectory_seeds": seeds,
        "trajectories": completed,
    }
    _write(root / "campaign_summary.json", summary)
    (root / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    return summary


def _audit(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.action == "prepare-open-refits":
        return prepare_open_refits(
            task_id=_task_id(args.task),
            discovery_root=args.discovery_root,
            output_root=args.output_root,
            source_trajectory=args.source_trajectory,
            seed_base=args.seed_base,
            device=args.device,
        )
    if args.action == "freeze":
        method = AUDIT_METHODS[args.method]
        return freeze_panel(
            method=method,
            task_id=_task_id(args.task),
            discovery_root=args.discovery_root,
            output_root=args.output_root,
            trajectory_count=OPEN_REFIT_COUNT if method == "open" else CONSTRAINED_ENDPOINT_COUNT,
        )
    if args.action == "run":
        return run_all(output_root=args.output_root, fold=args.fold, device=args.device)
    if args.action == "summarize":
        return summarize(output_root=args.output_root, fold=args.fold)
    if args.action == "authorize-fold5":
        return freeze_fold5(output_root=args.output_root)
    raise ValueError(f"unknown audit action: {args.action}")


def _validate(_args: argparse.Namespace) -> Mapping[str, Any]:
    runtime = load_runtime_config("configs/joint_response_runtime.json")
    starting_candidate = load_starting_candidate("h0", runtime)
    tasks: dict[str, Any] = {}
    for alias, task_id in TASKS.items():
        task = get_task(task_id)
        sources = {name: project_path(path).is_file() for name, path in task["sources"].items()}
        tasks[alias] = {"task_id": task_id, "sources": sources, "ready": all(sources.values())}
    return {
        "status": "valid",
        "task_registry": tasks,
        "starting_candidate_hash": starting_candidate.candidate_hash,
        "fold_roles": runtime["fold_roles"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cellaudit", description="CellAudit discover-then-falsify toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    discover = sub.add_parser("discover", help="run one discovery trajectory")
    discover.add_argument(
        "--stage",
        choices=tuple(STAGES),
        required=True,
        help=(
            "search setting: open=prediction-score discovery; "
            "source-constrained=Path-constrained discovery; "
            "falsification-guided=falsification-guided discovery"
        ),
    )
    discover.add_argument("--task", choices=tuple(TASKS), required=True)
    discover.add_argument("--output-root", required=True)
    discover.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    discover.add_argument("--device", default="cuda:0")
    discover.add_argument("--seed", type=int, default=2026080701)
    discover.add_argument("--slots", type=int)
    discover.set_defaults(handler=_discover)

    campaign = sub.add_parser("campaign", help="run the registered ten-trajectory campaign")
    campaign.add_argument(
        "--stage",
        choices=tuple(STAGES),
        required=True,
        help=(
            "search setting: open=prediction-score discovery; "
            "source-constrained=Path-constrained discovery; "
            "falsification-guided=falsification-guided discovery"
        ),
    )
    campaign.add_argument("--task", choices=tuple(TASKS), required=True)
    campaign.add_argument("--output-root", required=True)
    campaign.add_argument("--device", default="cuda:0")
    campaign.add_argument("--seed-base", type=int, default=2026080701)
    campaign.add_argument("--trajectories", type=int, default=10)
    campaign.set_defaults(handler=_campaign)

    audit = sub.add_parser("audit", help="freeze and execute deterministic held-out audits")
    audit.add_argument("action", choices=("prepare-open-refits", "freeze", "run", "summarize", "authorize-fold5"))
    audit.add_argument("--method", choices=tuple(AUDIT_METHODS))
    audit.add_argument("--task", choices=tuple(TASKS))
    audit.add_argument("--discovery-root")
    audit.add_argument("--output-root", required=True)
    audit.add_argument("--fold", type=int, choices=(4, 5), default=4)
    audit.add_argument("--device", default="cuda:0")
    audit.add_argument(
        "--source-trajectory",
        type=int,
        help="compatibility override for a pre-registered trajectory; by default the method-level Fold-3 winner is selected automatically",
    )
    audit.add_argument("--seed-base", type=int, default=2026080901)
    audit.set_defaults(handler=_audit)

    validate = sub.add_parser("validate", help="validate configuration and data bindings")
    validate.set_defaults(handler=_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "audit" and args.action == "freeze":
        if not args.method or not args.task or not args.discovery_root:
            raise SystemExit("audit freeze requires --method, --task, and --discovery-root")
    if args.command == "audit" and args.action == "prepare-open-refits":
        if not args.task or not args.discovery_root:
            raise SystemExit("audit prepare-open-refits requires --task and --discovery-root")
    result = args.handler(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
