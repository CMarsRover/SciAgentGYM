"""
Dataset configuration utilities for the testall suite.

Registers the two benchmark JSON files actually shipped with the repository:

- `dataset/refine_merged_multi_questions.json`  (multi-modal, 83 cases; DEFAULT)
- `dataset/refine_merged_single_questions.json` (single-modal, 48 cases)

The historic keys (``merged_questions_augmented`` / ``merged_single_questions``)
are kept as aliases so old scripts / trace files keep resolving.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DatasetEntry:
    """Lightweight descriptor holding dataset JSON and trace directory."""

    __slots__ = ("key", "label", "dataset_path", "trace_root")

    def __init__(self, key: str, label: str, dataset_path: Path, trace_root: Path):
        self.key = key
        self.label = label
        self.dataset_path = dataset_path
        self.trace_root = trace_root

    def to_dict(self) -> Dict[str, str]:
        return {
            "key": self.key,
            "label": self.label,
            "dataset_path": str(self.dataset_path),
            "trace_root": str(self.trace_root),
        }


_TRACE_ROOT = PROJECT_ROOT / "data_analysis" / "tracetoanalyze" / "traces"

_MULTI_ENTRY = DatasetEntry(
    key="refine_merged_multi_questions",
    label="refine_merged_multi_questions (multi-modal, 83 cases)",
    dataset_path=PROJECT_ROOT / "dataset" / "refine_merged_multi_questions.json",
    trace_root=_TRACE_ROOT,
)
_SINGLE_ENTRY = DatasetEntry(
    key="refine_merged_single_questions",
    label="refine_merged_single_questions (single-modal, 48 cases)",
    dataset_path=PROJECT_ROOT / "dataset" / "refine_merged_single_questions.json",
    trace_root=_TRACE_ROOT,
)

_DATASETS: Mapping[str, DatasetEntry] = {
    _MULTI_ENTRY.key: _MULTI_ENTRY,
    _SINGLE_ENTRY.key: _SINGLE_ENTRY,
    # Short aliases for CLI ergonomics.
    "multi": _MULTI_ENTRY,
    "single": _SINGLE_ENTRY,
    # Legacy keys used by older scripts and trace files — kept for
    # back-compat so nothing silently mismatches.
    "merged_questions_augmented": _MULTI_ENTRY,
    "merged_single_questions": _SINGLE_ENTRY,
}

DEFAULT_DATASET_KEY = _MULTI_ENTRY.key
_current_dataset_key: str = DEFAULT_DATASET_KEY


def list_available_datasets() -> Iterable[DatasetEntry]:
    """Yield every canonical registered dataset entry (no aliases)."""
    seen: set[str] = set()
    unique: list[DatasetEntry] = []
    for entry in _DATASETS.values():
        if entry.key in seen:
            continue
        seen.add(entry.key)
        unique.append(entry)
    return unique


def get_dataset_entry(dataset_key: str | None = None) -> DatasetEntry:
    """Resolve the dataset entry for the given (or active) key."""
    key = dataset_key or _current_dataset_key
    if key not in _DATASETS:
        raise ValueError(f"Unsupported dataset key: {key}")
    return _DATASETS[key]


def set_current_dataset_key(dataset_key: str) -> DatasetEntry:
    """Set the active dataset key and return the resolved entry."""
    global _current_dataset_key
    entry = get_dataset_entry(dataset_key)
    _current_dataset_key = entry.key
    return entry


def get_current_dataset_key() -> str:
    """Return the currently active dataset key."""
    return _current_dataset_key


def get_dataset_path(dataset_key: str | None = None) -> Path:
    """Return the dataset JSON path for the requested (or active) key."""
    return get_dataset_entry(dataset_key).dataset_path


def get_trace_root(dataset_key: str | None = None) -> Path:
    """Return the trace root directory for the requested (or active) key."""
    return get_dataset_entry(dataset_key).trace_root


def ensure_trace_root_exists(dataset_key: str | None = None) -> Path:
    """Ensure the trace directory exists for the dataset and return it."""
    trace_root = get_trace_root(dataset_key)
    trace_root.mkdir(parents=True, exist_ok=True)
    return trace_root
