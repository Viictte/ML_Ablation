"""Caching utilities for SpikePointNet experiment notebooks."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

ARTIFACT_ROOT = Path("../artifacts/spikepointnet")
HISTORY_DIR = ARTIFACT_ROOT / "histories"
CHECKPOINT_DIR = ARTIFACT_ROOT / "checkpoints"


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _make_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _make_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_serializable(v) for v in value]
    if hasattr(value, "__dict__"):
        return _make_serializable(vars(value))
    return value


def _normalize_history(history: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for record in history:
        entry: Dict[str, Any] = {str(k): _make_serializable(v) for k, v in dict(record).items()}
        test_value = entry.get("test_instance_acc")
        if test_value is not None and "test_acc" not in entry:
            entry["test_acc"] = test_value
        normalized.append(entry)
    return normalized


def save_training_history(
    history: Sequence[Mapping[str, Any]],
    experiment_name: str,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Persist a training history to the artifacts directory."""
    normalized = _normalize_history(history)
    payload: Dict[str, Any] = {"history": normalized}
    if metadata:
        payload["metadata"] = _make_serializable(metadata)

    _ensure_directory(HISTORY_DIR)
    path = HISTORY_DIR / f"{experiment_name}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def load_training_history(
    experiment_name: str,
    *,
    with_metadata: bool = False,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    """Load a cached training history if it exists."""
    path = HISTORY_DIR / f"{experiment_name}.json"
    if not path.exists():
        return (None, None) if with_metadata else (None, None)

    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    history = payload.get("history", [])
    metadata = payload.get("metadata")
    if with_metadata:
        return history, metadata
    return history, None


def cache_checkpoint(source: Path, experiment_name: str) -> Path:
    """Copy a checkpoint into the shared cache directory."""
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {source_path}")

    _ensure_directory(CHECKPOINT_DIR)
    destination = CHECKPOINT_DIR / f"{experiment_name}.pth"
    shutil.copy2(source_path, destination)
    return destination


def load_cached_checkpoint_path(experiment_name: str) -> Optional[Path]:
    """Return the cached checkpoint path for an experiment if available."""
    path = CHECKPOINT_DIR / f"{experiment_name}.pth"
    return path if path.exists() else None


def best_metric(history: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Optional[float]:
    """Return the best (maximum) metric value across the history for the given keys."""
    best_value: Optional[float] = None
    for record in history:
        for key in keys:
            value = record.get(key)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if best_value is None or numeric > best_value:
                best_value = numeric
            break
    return best_value


__all__ = [
    "save_training_history",
    "load_training_history",
    "cache_checkpoint",
    "load_cached_checkpoint_path",
    "best_metric",
]
