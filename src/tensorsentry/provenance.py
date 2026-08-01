"""Model provenance verification — **m3 milestone (stub)**.

The full provenance pass (m3) fetches the ModelScope / Hugging Face model card
and signed manifest, and verifies the sigstore bundle attaching the manifest
to the published artifact. This is **not implemented in v0.1** — m1 (structure)
and m2 (exploit) ship fully; provenance is the m3 milestone.

The stub raises :class:`ProvenanceNotImplemented` so callers (and the CLI's
``--provenance`` flag) fail loudly with a clear pointer to the milestone rather
than silently returning a fake ``signed`` verdict. When m3 lands, replace this
file's body with the real ``requests`` + ``sigstore`` implementation; the
:class:`ProvenanceResult` shape and :func:`verify_provenance` signature below
are the intended stable surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "ProvenanceResult",
    "ProvenanceNotImplemented",
    "verify_provenance",
    "M3_TODO",
]

M3_TODO = (
    "m3 milestone: provenance (ModelScope/HF manifest fetch + sigstore verify). "
    "Wire requests -> GET model card + signed-manifest; sigstore.verify.Verifier "
    "to check the bundle. See mvp_plan.md milestone m3_provenance_release."
)


class ProvenanceNotImplemented(NotImplementedError):
    """Raised when provenance is requested before the m3 milestone lands."""


@dataclass
class ProvenanceResult:
    """Shape of the m3 provenance verdict (returned by verify_provenance once m3 lands)."""

    provenance: str = "unreachable"  # "signed" | "unsigned" | "unreachable"
    model_repo: str = ""
    manifest_url: str = ""
    signer: str = ""
    anomalies: list[str] = field(default_factory=list)


def verify_provenance(path: str, model_id: str | None = None) -> ProvenanceResult:
    """Verify a weight artifact's ModelScope/HF signed-manifest provenance.

    .. note:: m3 stub — not yet implemented. Raises
       :class:`ProvenanceNotImplemented`. Implement in m3:

       1. Resolve the model repo (ModelScope ``model_id`` -> repo id, or HF
          fallback) from the artifact path / ``--model`` flag.
       2. ``requests.get`` the model card + ``*.signed-manifest.json``.
       3. ``sigstore.verify.Verifier`` the bundle; on success -> ``signed``;
          missing manifest -> ``unsigned``; network failure -> ``unreachable``.
    """
    raise ProvenanceNotImplemented(M3_TODO)
