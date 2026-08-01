"""Per-model ``TensorProfile`` registry.

Adding a new CN model = one new file in this package exposing a ``PROFILE``
attribute and registering it via ``register()``. Profiles are pure-data modules
so the structural schemas are auditable and cheap to extend.
"""

from __future__ import annotations

from ..tensor_validate import TensorProfile
from . import deepseek_v4, kimi_k3, qwen3_7

__all__ = ["PROFILES", "get_profile", "register", "list_profiles"]

# Registry of model_id -> TensorProfile.
PROFILES: dict[str, TensorProfile] = {}


def register(profile: TensorProfile) -> TensorProfile:
    """Register a profile by its ``model_id``."""
    PROFILES[profile.model_id] = profile
    return profile


# Built-in profiles ship registered on import.
register(deepseek_v4.PROFILE)
register(kimi_k3.PROFILE)
register(qwen3_7.PROFILE)


def get_profile(model_id: str) -> TensorProfile | None:
    """Look up a profile by id (e.g. ``deepseek-v4``). Returns None if unknown."""
    return PROFILES.get(model_id)


def list_profiles() -> list[TensorProfile]:
    """All registered profiles, sorted by id."""
    return sorted(PROFILES.values(), key=lambda p: p.model_id)
