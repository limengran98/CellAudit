"""Small shared contract for candidate-controlled training settings.

Open discovery can describe arbitrary architectures in candidate *source*, but
only this short, executor-owned set of top-level JSON keys has a runtime
meaning. The compiler uses the same definition when deciding whether a
configuration-only revision is a scientific action, preventing descriptive
``architecture``/``optimizer`` metadata from being mistaken for an executable
change.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


RUNTIME_TRAINING_CONTROL_KEYS = frozenset(
    {
        "batch_size",
        "max_epochs",
        "patience",
        "gradient_clip",
        "gradient_clip_norm",
        "min_delta",
    }
)


def _leaf_paths(
    value: Mapping[str, Any],
    prefix: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], ...]:
    paths: list[tuple[str, ...]] = []
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        path = (*prefix, key)
        if isinstance(item, Mapping) and item:
            paths.extend(_leaf_paths(item, path))
        else:
            paths.append(path)
    return tuple(paths)


def runtime_control_paths(patch: Mapping[str, Any]) -> tuple[tuple[str, ...], ...]:
    """Return configuration leaves consumed by the fixed training runtime."""

    if not isinstance(patch, Mapping):
        raise TypeError("training configuration patch must be a mapping")
    return tuple(
        path
        for path in _leaf_paths(patch)
        if len(path) == 1 and path[0] in RUNTIME_TRAINING_CONTROL_KEYS
    )


def non_runtime_control_paths(
    patch: Mapping[str, Any],
) -> tuple[tuple[str, ...], ...]:
    """Return every patch leaf that has no executor-defined semantics."""

    if not isinstance(patch, Mapping):
        raise TypeError("training configuration patch must be a mapping")
    return tuple(
        path
        for path in _leaf_paths(patch)
        if len(path) != 1 or path[0] not in RUNTIME_TRAINING_CONTROL_KEYS
    )


def require_runtime_control_patch(
    patch: Mapping[str, Any],
    *,
    name: str = "training configuration patch",
) -> None:
    """Fail closed when a formal run receives descriptive config metadata."""

    rejected = non_runtime_control_paths(patch)
    if rejected:
        rendered = [".".join(path) for path in rejected]
        raise ValueError(
            f"{name} contains non-executable configuration paths: {rendered}; "
            "architecture, loss, optimizer, and scheduler choices must be in candidate source"
        )


def has_runtime_control_effect(patch: Mapping[str, Any]) -> bool:
    """Whether a JSON merge patch changes any executor-consumed field."""

    return bool(runtime_control_paths(patch))


__all__ = [
    "RUNTIME_TRAINING_CONTROL_KEYS",
    "has_runtime_control_effect",
    "non_runtime_control_paths",
    "require_runtime_control_patch",
    "runtime_control_paths",
]
