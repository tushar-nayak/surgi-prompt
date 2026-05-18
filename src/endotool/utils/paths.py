from __future__ import annotations

from pathlib import Path


def grounding_dino_default_config() -> str:
    import groundingdino

    root = Path(groundingdino.__file__).resolve().parent
    return str(root / "config" / "GroundingDINO_SwinT_OGC.py")


def sam2_default_config() -> str:
    import sam2

    repo_root = Path(sam2.__file__).resolve().parent.parent
    return str(repo_root / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml")
