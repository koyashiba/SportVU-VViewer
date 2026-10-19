"""Compute backend selection: CUDA (NVIDIA), MPS (Apple Silicon) or CPU.

Auto-detection never imports torch unless it is installed; the detector
modules consume the resolved device string.
"""
from __future__ import annotations

from dataclasses import dataclass

BACKENDS = ("auto", "cuda", "mps", "cpu")


@dataclass(frozen=True)
class ComputeBackend:
    requested: str
    resolved: str
    reason: str


def resolve_backend(requested: str = "auto") -> ComputeBackend:
    requested = (requested or "auto").lower()
    if requested not in BACKENDS:
        raise ValueError(f"unknown compute backend {requested!r}; expected one of {BACKENDS}")
    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover - depends on environment
        if requested in ("cuda", "mps"):
            return ComputeBackend(requested, "cpu", f"{requested} requested but torch is not installed; using cpu")
        return ComputeBackend(requested, "cpu", "torch not installed; using cpu")
    cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    mps_ok = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if requested == "cuda":
        return ComputeBackend(requested, "cuda" if cuda_ok else "cpu", "cuda available" if cuda_ok else "cuda requested but unavailable; using cpu")
    if requested == "mps":
        return ComputeBackend(requested, "mps" if mps_ok else "cpu", "mps available" if mps_ok else "mps requested but unavailable; using cpu")
    if requested == "cpu":
        return ComputeBackend(requested, "cpu", "cpu requested")
    if cuda_ok:
        return ComputeBackend(requested, "cuda", "auto: cuda available")
    if mps_ok:
        return ComputeBackend(requested, "mps", "auto: mps available")
    return ComputeBackend(requested, "cpu", "auto: no accelerator found")
