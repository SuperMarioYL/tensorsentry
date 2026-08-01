"""Header-only ``.safetensors`` reader.

The safetensors on-disk format is a length-prefixed JSON header followed by the
raw tensor bytes::

    +--------------------+-------------------------------------------+
    | u64 LE header_len  | JSON: {tensor_name: {dtype,shape,          |
    | (8 bytes)           |         data_offsets:[start,end]}, ...}   |
    +--------------------+-------------------------------------------+
    | raw tensor data (never read by TensorSentry)                     |
    +-------------------------------------------------------------------+

TensorSentry only ever parses the header — names, dtypes and shapes — so a 70 GB
checkpoint scans in seconds without loading weights. This is verified against a
real shard (see the plan's schema smoke): an 8-byte little-endian u64 length
prefix, then a JSON object mapping tensor name to ``{dtype, shape, data_offsets}``
plus an optional top-level ``__metadata__`` key.

The reader is fully self-contained (no ``safetensors`` runtime dependency) so it
works in any environment. If the ``safetensors`` library happens to be
installed, it is *not* required — the byte-level parse is the source of truth.
"""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any, Iterator

__all__ = [
    "TensorInfo",
    "SafetensorsHeader",
    "read_header",
    "read_directory",
    "is_safetensors",
    "SafetensorsError",
]

# Magic sanity bounds — header length is u64, but a sane safetensors header is
# well under a few hundred MB. Anything larger is almost certainly not a
# safetensors file (or a maliciously crafted one we should refuse to parse).
_MAX_HEADER_BYTES = 512 * 1024 * 1024  # 512 MiB — generous for sharded 70B+ ckpts

_LAYER_RE = re.compile(r"layers\.(\d+)\.")


class SafetensorsError(Exception):
    """Raised when a file claims to be safetensors but its header is invalid."""


@dataclass(frozen=True)
class TensorInfo:
    """One tensor entry from a safetensors header."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def nbytes(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]


@dataclass
class SafetensorsHeader:
    """Parsed safetensors header for one shard (or one single-file checkpoint)."""

    path: str
    tensors: dict[str, TensorInfo] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    data_offset: int = 0  # absolute byte offset where the tensor data section begins
    format: str = "safetensors"

    @property
    def tensor_names(self) -> list[str]:
        return list(self.tensors.keys())

    def __iter__(self) -> Iterator[TensorInfo]:
        return iter(self.tensors.values())

    def layer_index(self, name: str) -> int | None:
        """Extract the transformer layer index from a tensor name, if present."""
        m = _LAYER_RE.search(name)
        return int(m.group(1)) if m else None


def is_safetensors(path: str) -> bool:
    """Cheap extension/magic check — does ``path`` look like a safetensors file?"""
    if not os.path.isfile(path):
        return False
    if path.endswith(".safetensors") or path.endswith(".safetensors.json"):
        return True
    # Magic-byte check: a real safetensors file's first 8 bytes are a u64 LE
    # length whose decoded value, plus 8, is <= file size and >= 2 (min JSON "{}").
    try:
        size = os.path.getsize(path)
        if size < 10:
            return False
        with open(path, "rb") as f:
            head = f.read(8)
        (n,) = struct.unpack("<Q", head)
        return 2 <= n <= _MAX_HEADER_BYTES and n + 8 <= size
    except (OSError, struct.error):
        return False


def read_header(path: str) -> SafetensorsHeader:
    """Parse a single ``.safetensors`` file's header (no weight data loaded)."""
    if not os.path.isfile(path):
        raise SafetensorsError(f"not a file: {path}")

    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) < 8:
            raise SafetensorsError(f"file too small to be safetensors: {path}")
        (header_len,) = struct.unpack("<Q", prefix)
        if header_len < 2:
            raise SafetensorsError(f"invalid safetensors header length {header_len}: {path}")
        if header_len > _MAX_HEADER_BYTES:
            raise SafetensorsError(
                f"safetensors header length {header_len} exceeds sanity bound: {path}"
            )
        raw = f.read(header_len)
        if len(raw) < header_len:
            raise SafetensorsError(f"truncated safetensors header: {path}")

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SafetensorsError(f"corrupt safetensors header JSON in {path}: {exc}") from exc

    if not isinstance(obj, dict):
        raise SafetensorsError(f"safetensors header is not a JSON object: {path}")

    tensors: dict[str, TensorInfo] = {}
    metadata: dict[str, Any] = {}
    for name, entry in obj.items():
        if name == "__metadata__":
            if isinstance(entry, dict):
                metadata.update(entry)
            continue
        if not isinstance(entry, dict):
            raise SafetensorsError(f"tensor '{name}' entry is not an object: {path}")
        try:
            dtype = str(entry["dtype"])
            shape = tuple(int(x) for x in entry["shape"])
            offsets = tuple(int(x) for x in entry["data_offsets"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SafetensorsError(
                f"tensor '{name}' missing dtype/shape/data_offsets: {path}"
            ) from exc
        if len(offsets) != 2 or offsets[1] < offsets[0]:
            raise SafetensorsError(f"tensor '{name}' has bad data_offsets {offsets}: {path}")
        tensors[name] = TensorInfo(name=name, dtype=dtype, shape=shape, data_offsets=offsets)

    return SafetensorsHeader(
        path=path,
        tensors=tensors,
        metadata=metadata,
        data_offset=8 + header_len,
    )


def _index_json(directory: str) -> dict[str, str] | None:
    """Read a ``model.safetensors.index.json`` if present, mapping tensor name -> shard path."""
    idx = os.path.join(directory, "model.safetensors.index.json")
    if not os.path.isfile(idx):
        return None
    try:
        with open(idx, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    mapping = data.get("weight_map") if isinstance(data, dict) else None
    if not isinstance(mapping, dict):
        return None
    return {str(k): os.path.join(directory, str(v)) for k, v in mapping.items()}


def read_directory(directory: str) -> list[SafetensorsHeader]:
    """Parse every ``.safetensors`` shard in a model directory (header-only).

    If a ``model.safetensors.index.json`` weight map is present it is used to
    attribute tensors to shards; otherwise every ``*.safetensors`` file in the
    directory is parsed. This is how a real ModelScope pull (multi-shard
    DeepSeek-V4) is scanned in one shot without loading weights.
    """
    if not os.path.isdir(directory):
        raise SafetensorsError(f"not a directory: {directory}")

    index = _index_json(directory)
    if index is not None:
        shards: list[SafetensorsHeader] = []
        seen: set[str] = set()
        for shard_name in sorted(set(index.values())):
            if shard_name in seen or not os.path.isfile(shard_name):
                continue
            seen.add(shard_name)
            shards.append(read_header(shard_name))
        return shards

    # No index — parse every shard file directly.
    out: list[SafetensorsHeader] = []
    for name in sorted(os.listdir(directory)):
        if name.endswith(".safetensors"):
            out.append(read_header(os.path.join(directory, name)))
    return out


def merge(headers: list[SafetensorsHeader]) -> SafetensorsHeader:
    """Merge several shard headers into one virtual header (names may overlap across shards)."""
    merged = SafetensorsHeader(path="<merged>", data_offset=0)
    for h in headers:
        merged.tensors.update(h.tensors)
        merged.metadata.update(h.metadata)
    return merged
