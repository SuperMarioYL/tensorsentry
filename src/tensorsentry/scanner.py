"""Scan orchestrator — ties the readers, validator, pickle scanner and provenance
into a single :class:`~tensorsentry.report.ScanReport`.

One process, one pass. The readers never load full weights (header/metadata
only), so a 70 GB checkpoint scans in seconds.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

from . import gguf_reader, safetensors_reader
from .model_profiles import get_profile, list_profiles
from .pickle_scan import PickleResult, scan_path
from .provenance import ProvenanceNotImplemented, verify_provenance
from .report import ScanReport
from .tensor_validate import Anomaly, StructureResult, TensorView, TensorProfile, validate_structure

__all__ = ["scan", "validate", "load_tensors", "detect_source_format", "SourceFormatError"]


class SourceFormatError(Exception):
    """Raised when a path is neither a safetensors file/dir nor a gguf file."""


@dataclass(frozen=True)
class LoadedTensors:
    tensors: list[TensorView]
    source_format: str  # "safetensors" | "gguf"
    files: list[str]


def detect_source_format(path: str) -> str | None:
    """Best-effort detection: safetensors, gguf, or None."""
    if os.path.isdir(path):
        names = os.listdir(path)
        # Deterministic priority: safetensors (the structural-validation target
        # for sharded HF-style checkpoints) wins over gguf when both are present.
        if any(name.endswith(".safetensors") for name in names):
            return "safetensors"
        if os.path.isfile(os.path.join(path, "model.safetensors.index.json")):
            return "safetensors"
        if any(name.endswith(".gguf") for name in names):
            return "gguf"
        return None
    if safetensors_reader.is_safetensors(path) or path.endswith(".safetensors"):
        return "safetensors"
    if gguf_reader.is_gguf(path) or path.endswith(".gguf"):
        return "gguf"
    return None


def load_tensors(path: str) -> LoadedTensors:
    """Header-only load of every tensor in ``path`` (file or directory)."""
    fmt = detect_source_format(path)
    if fmt == "safetensors":
        if os.path.isdir(path):
            headers = safetensors_reader.read_directory(path)
            merged = safetensors_reader.merge(headers)
            tensors = list(merged.tensors.values())
            files = [h.path for h in headers]
        else:
            h = safetensors_reader.read_header(path)
            tensors = list(h.tensors.values())
            files = [h.path]
        return LoadedTensors(tensors=tensors, source_format="safetensors", files=files)
    if fmt == "gguf":
        if os.path.isdir(path):
            headers = [
                gguf_reader.read_header(os.path.join(path, name))
                for name in sorted(os.listdir(path))
                if name.endswith(".gguf")
            ]
            merged = gguf_reader.merge(headers)
            return LoadedTensors(
                tensors=list(merged.tensors.values()),
                source_format="gguf",
                files=[h.path for h in headers],
            )
        h = gguf_reader.read_header(path)
        return LoadedTensors(tensors=list(h.tensors.values()), source_format="gguf", files=[h.path])
    raise SourceFormatError(f"cannot determine weight format for: {path}")


def _run_structure(
    tensors: Iterable[TensorView], source_format: str, model_id: str | None
) -> tuple[StructureResult | None, str]:
    """Validate tensors against the named profile; returns (result, model_label)."""
    if not model_id:
        return None, "(no profile)"
    profile = get_profile(model_id)
    if profile is None:
        return StructureResult(profile=model_id, structure="unknown_profile"), model_id
    return validate_structure(tensors, profile, source_format), profile.model_id


def _reader_error_structure(model_label: str, exc: Exception) -> StructureResult:
    """A structure verdict for an artifact that claims a known weight format
    but cannot be parsed (truncated / tampered / corrupt)."""
    return StructureResult(
        profile=model_label,
        structure="anomaly",
        anomalies=[Anomaly(code="reader_error", message=str(exc))],
    )


def validate(model_id: str, path: str) -> ScanReport:
    """m1 path: structure-only validation of one artifact against one profile."""
    try:
        loaded = load_tensors(path)
    except SourceFormatError as exc:
        # Not a weight container at all — no structural verdict is possible.
        return ScanReport(
            path=path, model=model_id, source_format="unknown",
            structure="unknown_profile", exploit="clean",
            anomalies=[f"reader error: {exc}"],
        )
    except (safetensors_reader.SafetensorsError, gguf_reader.GGUFError) as exc:
        # The artifact claims a known weight format but is unparseable —
        # corrupt or tampered, which is itself a structural anomaly.
        structure = _reader_error_structure(model_id, exc)
        return ScanReport.from_results(
            path=path, model=model_id, source_format="unknown",
            structure=structure, exploit=None, scanned_files=[path],
        )
    structure, model_label = _run_structure(loaded.tensors, loaded.source_format, model_id)
    return ScanReport.from_results(
        path=path, model=model_label, source_format=loaded.source_format,
        structure=structure, exploit=None, scanned_files=loaded.files,
    )


def scan(
    path: str,
    model_id: str | None = None,
    include_provenance: bool = False,
) -> ScanReport:
    """m2 path: combined structure + exploit (+ optional provenance) scan."""
    try:
        loaded = load_tensors(path)
    except SourceFormatError:
        # Unknown container — still run the exploit scan: the file may be a
        # mislabelled pickle (a real attack shape).
        exploit: PickleResult = scan_path(path)
        return ScanReport.from_results(
            path=path, model=model_id or "(unknown)", source_format="unknown",
            structure=None, exploit=exploit, scanned_files=[path],
            provenance="unreachable",
        )
    except (safetensors_reader.SafetensorsError, gguf_reader.GGUFError) as exc:
        # Claims a known weight format but unparseable — corrupt or tampered.
        # Surface it as a structure anomaly (non-zero exit) while still
        # running the exploit pass.
        structure = _reader_error_structure(model_id or "(unknown)", exc)
        exploit_result = scan_path(path)
        return ScanReport.from_results(
            path=path, model=model_id or "(unknown)", source_format="unknown",
            structure=structure, exploit=exploit_result, scanned_files=[path],
            provenance="unreachable",
        )

    structure, model_label = _run_structure(loaded.tensors, loaded.source_format, model_id)
    exploit_result = scan_path(path)

    provenance = "unreachable"
    if include_provenance:
        try:
            prov = verify_provenance(path, model_id)
            provenance = prov.provenance
        except ProvenanceNotImplemented:
            provenance = "unreachable"

    return ScanReport.from_results(
        path=path,
        model=model_label,
        source_format=loaded.source_format,
        structure=structure,
        exploit=exploit_result,
        scanned_files=loaded.files,
        provenance=provenance,
    )


def known_profiles() -> list[TensorProfile]:
    return list_profiles()
