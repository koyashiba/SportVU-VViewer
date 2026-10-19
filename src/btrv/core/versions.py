"""Version information recorded in every run manifest for reproducibility."""
from __future__ import annotations

import platform
import sys
from importlib import metadata


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def software_versions() -> dict[str, str | None]:
    from btrv import __version__

    return {
        "btrv": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": package_version("numpy"),
        "opencv-python-headless": package_version("opencv-python-headless") or package_version("opencv-python"),
        "av": package_version("av"),
        "supervision": package_version("supervision"),
        "inference-models": package_version("inference-models"),
        "torch": package_version("torch"),
        "PySide6": package_version("PySide6"),
    }
