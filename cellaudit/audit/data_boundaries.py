"""Target-fold loaders used only after the corresponding audit authorization."""

from __future__ import annotations

from typing import Any, Mapping

from cellaudit.joint_response_runtime import (
    CPGFold14Arrays,
    CPGPartition,
    _decode,
    _manifest_verified_cache,
    _require_numpy,
)


class AuditDataBoundaryError(RuntimeError):
    """Raised when a registered held-out fold cannot be resolved exactly."""


def load_fold5_arrays(task_id: str) -> CPGFold14Arrays:
    """Open Fold 5 only from an explicitly authorized audit call."""

    np = _require_numpy()
    cache, _task, manifest, data_fingerprint = _manifest_verified_cache(task_id)
    with np.load(cache, allow_pickle=False) as archive:
        folds = archive["fold"].astype(np.int8, copy=False)
        labels = _decode(archive["labels"])
        raw = {
            name: archive[name].astype(np.float32, copy=False)
            for name in ("pre", "condition", "target", "cp_response", "l1000_response")
        }
    fit_indices = np.flatnonzero(np.isin(folds, [0, 1]))
    selection_indices = np.flatnonzero(folds == 2)
    endpoint_indices = np.flatnonzero(folds == 4)
    if not len(fit_indices) or not len(selection_indices) or not len(endpoint_indices):
        raise AuditDataBoundaryError("registered Fold-5 role partition is empty")

    def partition(indices: Any) -> CPGPartition:
        return CPGPartition(
            pre=raw["pre"][indices],
            condition=raw["condition"][indices],
            target=raw["target"][indices],
            cp_target=raw["cp_response"][indices],
            l1000_target=raw["l1000_response"][indices],
            labels=tuple(labels[int(index)] for index in indices.tolist()),
            cache_folds=folds[indices].copy(),
        )

    fit, selection, endpoint = partition(fit_indices), partition(selection_indices), partition(endpoint_indices)
    if {int(value) for value in endpoint.cache_folds.tolist()} != {4}:
        raise AuditDataBoundaryError("endpoint role does not resolve to Fold 5")
    layout = manifest.get("condition_layout")
    if not isinstance(layout, Mapping):
        raise AuditDataBoundaryError("registered condition layout is absent")
    return CPGFold14Arrays(
        task_id=task_id,
        fit=fit,
        selection=selection,
        endpoint=endpoint,
        condition_layout=dict(layout),
        data_fingerprint=data_fingerprint,
        metadata={
            "task_id": task_id,
            "data_fingerprint": data_fingerprint,
            "loaded_target_folds": [1, 2, 3, 5],
            "excluded_target_folds": [4],
            "fit_rows": fit.count,
            "selection_rows": selection.count,
            "endpoint_rows": endpoint.count,
            "condition_layout": dict(layout),
        },
    )
