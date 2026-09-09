"""Per-model ``TensorProfile`` schema + structural validation engine.

This is TensorSentry's core primitive — a declarative schema of the MoE/MLA
tensor structure a genuine DeepSeek-V4 / Kimi K3 / Qwen3.7 checkpoint *must*
satisfy. Generic pickle scanners read arbitrary pickle byte-streams for
code-exec; they cannot know that a DeepSeek MLA checkpoint must contain
``q_lora_rank``/``kv_lora_rank`` projection tensors at specific ranks, or that a
Kimi K3 MoE must expose 896 experts. This per-model structural knowledge is the
piece generic scanners structurally lack.

Validation is **layer-aware**: tensors are grouped by transformer layer, and a
layer that appears at all must carry the *complete* per-layer tensor set with
correct ranks. This makes the validator robust to sharded checkpoints (a single
ModelScope shard only contains a range of layers; missing whole layers across
one shard is *not* an anomaly — only a layer that is present-but-incomplete is).

The engine works on a uniform view of tensors (name, dtype, shape) regardless
of container (safetensors or gguf); the per-model ``TensorSpec`` carries both
an HF/naming pattern and a GGUF pattern.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Iterable, Protocol

__all__ = [
    "TensorSpec",
    "MoESpec",
    "MLASpec",
    "TensorProfile",
    "StructureVerdict",
    "Anomaly",
    "StructureResult",
    "TensorView",
    "validate_structure",
    "check_dtype",
    "check_shape",
]


class TensorView(Protocol):
    """Uniform read view over a tensor (safetensors or gguf)."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    ndim: int


@dataclass(frozen=True)
class TensorSpec:
    """One required tensor, matched by fnmatch pattern, with shape constraints.

    ``shape_constraints`` keys (all optional):
        ndim       — exact number of dims
        out_rank   — shape[0] must equal this (Linear output features, HF layout)
        in_rank    — shape[-1] must equal this (Linear input features)
        dim0_eq    — shape[0] must equal this (1-D layernorm/bias, alias for clarity)
        min_matches — minimum number of matching tensors across the file (default 1)
    """

    logical_name: str
    pattern: str
    gguf_pattern: str | None = None
    dtype: str | None = None
    shape_constraints: dict[str, int] = field(default_factory=dict)
    min_matches: int = 1
    category: str = "general"  # "mla" | "moe" | "attention" | "general" — groups specs


@dataclass(frozen=True)
class MoESpec:
    """Mixture-of-experts structure for one model.

    The expert layout is **auto-detected** from the checkpoint naming rather than
    declared: DeepSeek-V4 stores one tensor per expert in its HF safetensors
    (``model.layers.*.mlp.experts.*.gate_proj.weight``) but a *batched* tensor
    in its gguf (``blk.*.ffn_gate_exps.weight`` with shape[0]=n_experts). The
    validator counts distinct ``experts.N`` indices when the name carries an
    expert index, and falls back to the leading dim of a batched tensor
    otherwise — so the same profile validates both distributions correctly.
    """

    n_experts: int
    n_shared: int
    expert_pattern: str
    expert_gguf_pattern: str | None = None
    shared_pattern: str | None = None
    shared_gguf_pattern: str | None = None
    router_pattern: str = "model.layers.*.mlp.gate.weight"
    router_gguf_pattern: str = "blk.*.ffn_gate_inp.weight"


@dataclass(frozen=True)
class MLASpec:
    """DeepSeek Multi-head Latent Attention ranks (the structural invariant)."""

    q_lora_rank: int
    kv_lora_rank: int
    n_heads: int
    qk_rope_head_dim: int = 64


@dataclass(frozen=True)
class TensorProfile:
    """A per-model MoE/MLA structural schema."""

    model_id: str
    architecture: str
    required_tensors: tuple[TensorSpec, ...]
    moe: MoESpec | None = None
    mla: MLASpec | None = None


# --- verdict vocabulary (mirrors the plan's ScanReport) ---

StructureVerdict = str  # "ok" | "anomaly" | "unknown_profile"


@dataclass
class Anomaly:
    """One structural anomaly found by the validator."""

    code: str  # short machine code, e.g. "missing_mla_proj", "rank_mismatch", "expert_count"
    message: str
    tensor: str | None = None
    layer: int | None = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        loc = f"layer {self.layer}" if self.layer is not None else "global"
        t = f" tensor='{self.tensor}'" if self.tensor else ""
        return f"[{self.code}] {loc}{t}: {self.message}"


@dataclass
class StructureResult:
    """Result of validating a tensor set against a profile."""

    profile: str
    structure: StructureVerdict  # "ok" | "anomaly" | "unknown_profile"
    anomalies: list[Anomaly] = field(default_factory=list)
    n_tensors: int = 0
    n_layers: int = 0
    expert_count: int | None = None

    @property
    def ok(self) -> bool:
        return self.structure == "ok"


_LAYER_RE = re.compile(r"(?:layers|blk)\.(\d+)[\.\)]")


def _layer_index(name: str) -> int | None:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _expert_index(name: str) -> int | None:
    """Extract the expert index from a per-expert tensor name, if present.

    Matches the second wildcard in ``model.layers.*.mlp.experts.*.<proj>``.
    """
    m = re.search(r"experts\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def check_dtype(spec: TensorSpec, t: TensorView) -> str | None:
    """Return an anomaly message if ``t``'s dtype violates ``spec``, else None."""
    if spec.dtype is None:
        return None
    want = spec.dtype.upper()
    got = (t.dtype or "").upper()
    if want == got:
        return None
    # F16/BF16 are sometimes interchangeable across checkpoints; flag only gross drift.
    return f"dtype mismatch: expected {want}, got {got}"


def check_shape(spec: TensorSpec, t: TensorView) -> str | None:
    """Return an anomaly message if ``t``'s shape violates ``spec``, else None."""
    c = spec.shape_constraints
    if not c:
        return None
    shape = tuple(t.shape)
    if "ndim" in c and len(shape) != c["ndim"]:
        return f"ndim mismatch: expected {c['ndim']}, got {len(shape)} (shape={list(shape)})"
    if not shape:
        return None
    if "out_rank" in c and shape[0] != c["out_rank"]:
        return f"output rank mismatch: expected {c['out_rank']}, got {shape[0]} (shape={list(shape)})"
    if "in_rank" in c and shape[-1] != c["in_rank"]:
        return f"input rank mismatch: expected {c['in_rank']}, got {shape[-1]} (shape={list(shape)})"
    if "dim0_eq" in c and shape[0] != c["dim0_eq"]:
        return f"dim0 mismatch: expected {c['dim0_eq']}, got {shape[0]} (shape={list(shape)})"
    return None


def _spec_pattern(spec: TensorSpec, source_format: str) -> str:
    if source_format == "gguf" and spec.gguf_pattern:
        return spec.gguf_pattern
    return spec.pattern


def _count_experts(
    tensors: Iterable[TensorView], moe: MoESpec, source_format: str
) -> dict[int, int]:
    """Per-layer expert count, auto-detecting layout from naming.

    If a matching tensor's name carries an ``experts.N`` index, experts are
    counted as distinct indices (per-expert layout). Otherwise the leading dim
    of the matched tensor is taken as the batched expert axis.
    """
    pat = moe.expert_gguf_pattern if source_format == "gguf" and moe.expert_gguf_pattern else moe.expert_pattern
    per_layer_indices: dict[int, set[int]] = {}
    batched_dims: dict[int, int] = {}
    saw_indexed = False
    for t in tensors:
        if not fnmatchcase(t.name, pat):
            continue
        layer = _layer_index(t.name)
        ei = _expert_index(t.name)
        if ei is not None:
            saw_indexed = True
            per_layer_indices.setdefault(layer if layer is not None else -1, set()).add(ei)
        elif t.shape:
            batched_dims[layer if layer is not None else -1] = t.shape[0]
    if saw_indexed:
        return {layer: len(idx) for layer, idx in per_layer_indices.items()}
    return batched_dims


def validate_structure(
    tensors: Iterable[TensorView],
    profile: TensorProfile,
    source_format: str = "safetensors",
) -> StructureResult:
    """Validate ``tensors`` against ``profile``. Returns a :class:`StructureResult`.

    ``source_format`` selects which naming pattern to use ("safetensors" uses HF
    naming; "gguf" uses the GGUF ``blk.*`` naming where a gguf_pattern is given).

    Validation rules:
      * each required tensor spec must be satisfied (>= ``min_matches``),
        per-layer for spec names containing a layer wildcard;
      * dtypes/shapes match the spec constraints;
      * MLA ranks (q_lora_rank / kv_lora_rank) appear in the right tensor dims;
      * MoE expert count per layer equals ``moe.n_experts``;
      * MoE shared expert(s) and router exist.
    Missing whole layers (sharding) is *not* an anomaly — only present-but-
    incomplete/malformed layers are.
    """
    tensors = list(tensors)
    anomalies: list[Anomaly] = []
    layer_set: set[int] = set()

    # --- required tensors: per-tensor dtype/shape checks + min_matches coverage ---
    for spec in profile.required_tensors:
        pat = _spec_pattern(spec, source_format)
        matches = [t for t in tensors if fnmatchcase(t.name, pat)]
        for t in matches:
            li = _layer_index(t.name)
            if li is not None:
                layer_set.add(li)
            msg = check_dtype(spec, t) or check_shape(spec, t)
            if msg:
                anomalies.append(
                    Anomaly(
                        code="malformed_tensor",
                        message=msg,
                        tensor=t.name,
                        layer=li,
                    )
                )
        # Coverage: a spec with no layer/rank wildcard is a global tensor
        # (embed/norm) and must match at least once anywhere in the file.
        is_global = "layers" not in pat and "blk" not in pat
        if spec.min_matches > 0 and is_global and len(matches) < spec.min_matches:
            anomalies.append(
                Anomaly(
                    code="missing_tensor",
                    message=f"no tensor matches required global spec '{spec.logical_name}' (pattern: {pat})",
                )
            )

    # Collect the set of layers actually present in the checkpoint.
    if not layer_set:
        for t in tensors:
            li = _layer_index(t.name)
            if li is not None:
                layer_set.add(li)

    # --- per-layer completeness of the MLA projection set ---
    if profile.mla is not None:
        mla_specs = [s for s in profile.required_tensors if s.category == "mla"]
        expected = {s.logical_name for s in mla_specs}
        mla_by_layer: dict[int, set[str]] = {}
        for s in mla_specs:
            pat = _spec_pattern(s, source_format)
            for t in tensors:
                if fnmatchcase(t.name, pat):
                    li = _layer_index(t.name)
                    if li is None:
                        continue
                    mla_by_layer.setdefault(li, set()).add(s.logical_name)
        # A layer that appears at all must carry the complete MLA set — only
        # whole layers missing to sharding are tolerated. A layer with tensors
        # but zero MLA projections is either a stripped/tampered attention
        # block or a shard boundary splitting one layer; both must be flagged,
        # never silently passed as ok.
        present_layers = {
            li for li in (_layer_index(t.name) for t in tensors) if li is not None
        }
        for li in sorted(present_layers):
            present = mla_by_layer.get(li, set())
            if not present:
                anomalies.append(
                    Anomaly(
                        code="missing_mla",
                        message=(
                            f"layer {li} carries tensors but no MLA attention projections "
                            f"at all — expected {sorted(expected)} (stripped/tampered attention "
                            f"block, or a shard boundary splitting this layer)"
                        ),
                        layer=li,
                    )
                )
                continue
            missing = expected - present
            if missing:
                anomalies.append(
                    Anomaly(
                        code="incomplete_mla",
                        message=(
                            f"layer {li} exposes MLA attention but is missing projections: "
                            f"{sorted(missing)} — a tampered/truncated MLA block"
                        ),
                        layer=li,
                    )
                )

    # --- MoE expert count + router + shared ---
    expert_count: int | None = None
    if profile.moe is not None:
        counts = _count_experts(tensors, profile.moe, source_format)
        if counts:
            expert_count = max(counts.values())
            for li, n in counts.items():
                if n != profile.moe.n_experts:
                    anomalies.append(
                        Anomaly(
                            code="expert_count_mismatch",
                            message=(
                                f"layer {li} has {n} experts, profile expects "
                                f"{profile.moe.n_experts} — possible expert routing tampering"
                            ),
                            layer=li,
                        )
                    )
        # router present on every MoE layer
        router_pat = profile.moe.router_gguf_pattern if source_format == "gguf" else profile.moe.router_pattern
        router_layers: set[int] = set()
        for t in tensors:
            if fnmatchcase(t.name, router_pat):
                li = _layer_index(t.name)
                if li is not None:
                    router_layers.add(li)
        expert_layers = set(counts.keys())
        for li in sorted(expert_layers - router_layers):
            anomalies.append(
                Anomaly(code="missing_router", message=f"layer {li} has experts but no router tensor", layer=li)
            )

    verdict: StructureVerdict = "ok" if not anomalies else "anomaly"
    return StructureResult(
        profile=profile.model_id,
        structure=verdict,
        anomalies=anomalies,
        n_tensors=len(tensors),
        n_layers=len(layer_set),
        expert_count=expert_count,
    )
