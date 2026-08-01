"""Header-only ``.gguf`` reader.

GGUF (``gguf``/``ggml``) is the other weight container DeepSeek/Kimi/Qwen ship.
Its on-disk layout is::

    +-----------------+-------------------------------------------+
    | magic  "GGUF"   | version u32 | n_tensors u64 | n_kv u64   |
    +-----------------+-------------------------------------------+
    | metadata key/value pairs (typed)                            |
    +-------------------------------------------------------------+
    | tensor infos: name, n_dims, dims[], dtype, offset u64      |
    +-------------------------------------------------------------+
    | padding to alignment | raw tensor data (never read here)   |
    +-------------------------------------------------------------+

TensorSentry parses only the metadata + tensor-info tables — names, shapes and
dtypes — so a quantized checkpoint scans without loading weights. The parser is
self-contained (no ``gguf`` runtime dependency); it implements the public GGUF
v3 spec directly so it runs anywhere Python does.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Iterator

__all__ = [
    "GGUFTensor",
    "GGUFHeader",
    "read_header",
    "is_gguf",
    "GGUFError",
]

_MAGIC = b"GGUF"
_MAGIC_LEGACY = (b"GGML", b"GGJT")  # older formats — read-only awareness

# GGUFValueType enum (stable since v3).
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32, _F32, _BOOL, _STRING, _ARRAY, _UINT64, _INT64, _F64 = range(13)

# GGML tensor dtype enum (subset that matters for structural checks).
_GGML_DTYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 24: "IQ3_XXS", 28: "I8", 29: "I16", 30: "I32", 31: "I64",
    32: "F64", 33: "IQ1_S", 34: "IQ4_NL", 35: "BF16", 36: "IQ3_S", 37: "IQ2_S",
    38: "IQ4_XS", 39: "I1", 40: "IQ2_M", 203: "TQ1_0", 204: "TQ2_0",
}


class GGUFError(Exception):
    """Raised when a file claims to be GGUF but its header is invalid."""


@dataclass(frozen=True)
class GGUFTensor:
    """One tensor info entry from a GGUF header."""

    name: str
    dtype: str
    dims: tuple[int, ...]
    offset: int

    @property
    def shape(self) -> tuple[int, ...]:
        # GGUF stores dims in reverse (nelem-first) order; expose as-is but
        # expose ``shape`` alias matching safetensors convention callers expect.
        return self.dims

    @property
    def ndim(self) -> int:
        return len(self.dims)


@dataclass
class GGUFHeader:
    """Parsed GGUF header for one file."""

    path: str
    tensors: dict[str, GGUFTensor] = field(default_factory=dict)
    fields: dict[str, Any] = field(default_factory=dict)
    version: int = 0
    alignment: int = 32
    format: str = "gguf"

    @property
    def tensor_names(self) -> list[str]:
        return list(self.tensors.keys())

    def __iter__(self) -> Iterator[GGUFTensor]:
        return iter(self.tensors.values())


def is_gguf(path: str) -> bool:
    """Does ``path`` look like a GGUF file (extension or magic)?"""
    if not os.path.isfile(path):
        return False
    if path.endswith(".gguf"):
        return True
    try:
        with open(path, "rb") as f:
            return f.read(4) in (_MAGIC, *_MAGIC_LEGACY)
    except OSError:
        return False


def _read(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise GGUFError(f"unexpected EOF reading {n} bytes")
    return data


def _read_u32(f: BinaryIO) -> int:
    return struct.unpack("<I", _read(f, 4))[0]


def _read_u64(f: BinaryIO) -> int:
    return struct.unpack("<Q", _read(f, 8))[0]


def _read_string(f: BinaryIO) -> str:
    n = _read_u64(f)
    raw = _read(f, n)
    return raw.decode("utf-8", errors="replace")


def _read_value(f: BinaryIO, vtype: int) -> Any:
    if vtype == _UINT8:
        return struct.unpack("<B", _read(f, 1))[0]
    if vtype == _INT8:
        return struct.unpack("<b", _read(f, 1))[0]
    if vtype == _UINT16:
        return struct.unpack("<H", _read(f, 2))[0]
    if vtype == _INT16:
        return struct.unpack("<h", _read(f, 2))[0]
    if vtype == _UINT32:
        return _read_u32(f)
    if vtype == _INT32:
        return struct.unpack("<i", _read(f, 4))[0]
    if vtype == _UINT64:
        return _read_u64(f)
    if vtype == _INT64:
        return struct.unpack("<q", _read(f, 8))[0]
    if vtype == _F32:
        return struct.unpack("<f", _read(f, 4))[0]
    if vtype == _F64:
        return struct.unpack("<d", _read(f, 8))[0]
    if vtype == _BOOL:
        return bool(struct.unpack("<B", _read(f, 1))[0])
    if vtype == _STRING:
        return _read_string(f)
    if vtype == _ARRAY:
        elem_type = _read_u32(f)
        count = _read_u64(f)
        return [_read_value(f, elem_type) for _ in range(count)]
    raise GGUFError(f"unsupported GGUF value type {vtype}")


def read_header(path: str) -> GGUFHeader:
    """Parse a single ``.gguf`` file's metadata + tensor-info table (no weight data loaded)."""
    if not os.path.isfile(path):
        raise GGUFError(f"not a file: {path}")

    with open(path, "rb") as fh:
        magic = _read(fh, 4)
        if magic != _MAGIC:
            raise GGUFError(
                f"not a GGUF file (magic={magic!r}): {path}"
            )
        version = _read_u32(fh)
        if version < 1 or version > 3:
            raise GGUFError(f"unsupported GGUF version {version}: {path}")
        n_tensors = _read_u64(fh)
        n_kv = _read_u64(fh)

        if n_kv > 10_000_000 or n_tensors > 100_000_000:
            raise GGUFError(f"implausible GGUF counts (kv={n_kv}, tensors={n_tensors}): {path}")

        fields: dict[str, Any] = {}
        for _ in range(n_kv):
            key = _read_string(fh)
            vtype = _read_u32(fh)
            fields[key] = _read_value(fh, vtype)

        tensors: dict[str, GGUFTensor] = {}
        for _ in range(n_tensors):
            name = _read_string(fh)
            n_dims = _read_u32(fh)
            if n_dims > 8:
                raise GGUFError(f"tensor '{name}' has implausible n_dims={n_dims}: {path}")
            dims = tuple(_read_u64(fh) for _d in range(n_dims))
            dtype_id = _read_u32(fh)
            offset = _read_u64(fh)
            dtype = _GGML_DTYPE_NAMES.get(dtype_id, f"DTYPE{dtype_id}")
            tensors[name] = GGUFTensor(name=name, dtype=dtype, dims=dims, offset=offset)

    alignment = int(fields.get("general.alignment", 32)) or 32
    return GGUFHeader(
        path=path,
        tensors=tensors,
        fields=fields,
        version=version,
        alignment=alignment,
    )
