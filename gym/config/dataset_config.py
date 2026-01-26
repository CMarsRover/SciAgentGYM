"""
Dataset configuration utilities for the testall suite.

Allows switching between the refined-data JSON sources and their corresponding
trace output directories so that the CLI and helper scripts can target either
`dataset/merged_questions_augmented.json` (default, traces/) or
`dataset/merged_single_questions.json` (traces_refine_single_questions/).
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


_DATASETS: Mapping[str, DatasetEntry] = {
    "merged_questions_augmented": DatasetEntry(
        key="merged_questions_augmented",
        label="merged_questions_augmented (默认)",
        dataset_path=PROJECT_ROOT / "dataset" / "merged_questions_augmented.json",
        trace_root=PROJECT_ROOT / "tracetoanalyze" / "traces",
    ),
    "merged_single_questions": DatasetEntry(
        key="merged_single_questions",
        label="merged_single_questions",
        dataset_path=PROJECT_ROOT / "dataset" / "merged_single_questions.json",
        trace_root=PROJECT_ROOT / "tracetoanalyze" / "traces",  # 统一使用 traces 目录，通过 dataset_folder 区分
    ),
}

DEFAULT_DATASET_KEY = "merged_questions_augmented"
_current_dataset_key: str = DEFAULT_DATASET_KEY


def list_available_datasets() -> Iterable[DatasetEntry]:
    """Yield every registered dataset entry."""
    return _DATASETS.values()


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
