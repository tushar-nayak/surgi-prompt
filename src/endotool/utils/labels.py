from __future__ import annotations

from pathlib import Path

import yaml


def load_label_map(path: str | Path) -> dict[str, str]:
    payload = yaml.safe_load(Path(path).read_text())
    canonical = payload.get("canonical_labels", {})
    alias_to_canonical: dict[str, str] = {}
    for canonical_name, aliases in canonical.items():
        alias_to_canonical[_norm(canonical_name)] = canonical_name
        for alias in aliases:
            alias_to_canonical[_norm(alias)] = canonical_name
    return alias_to_canonical


def normalize_label(label: str, alias_to_canonical: dict[str, str] | None) -> str:
    if not alias_to_canonical:
        return label.strip().lower()
    norm = _norm(label)
    if norm in alias_to_canonical:
        return alias_to_canonical[norm]
    for alias, canonical in alias_to_canonical.items():
        if alias in norm or norm in alias:
            return canonical
    return label.strip().lower()


def build_label_index(labels: list[str]) -> dict[str, int]:
    return {label: idx for idx, label in enumerate(sorted(set(labels)), start=1)}


def _norm(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").replace(".", " ").split())
