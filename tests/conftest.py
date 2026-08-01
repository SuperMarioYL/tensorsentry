"""Shared pytest fixtures — build tiny but structurally-faithful weight artifacts.

These fixtures write real ``.safetensors`` / ``.gguf`` / pickle files whose
*headers* match a given TensorSentry profile, so the validators and readers are
exercised against the genuine on-disk formats (no 70 GB download needed).
Header-only fixtures keep the data region tiny — only names/dtypes/shapes
matter for structural validation.
"""

from __future__ import annotations

import json
import os
import pickle
import struct
import sys
import tempfile
from pathlib import Path

import pytest

# --- safetensors fixture writer ------------------------------------------------

def _write_safetensors(path: str, tensors: dict, metadata: dict | None = None) -> str:
    """Write a minimal valid .safetensors file (header-only; tiny shared data region)."""
    # All tensors share a 1-byte data region — only the header (names/shapes/dtypes)
    # is load-bearing for structural validation, so a degenerate data section is fine.
    header = {}
    if metadata:
        header["__metadata__"] = metadata
    for name, (dtype, shape) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [0, 1]}
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # safetensors pads the header so (8 + header_len) is a multiple of 8.
    pad = (8 - (len(raw) % 8)) % 8
    raw_padded = raw + b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw_padded)))
        f.write(raw_padded)
        f.write(b"\x00")  # 1-byte data region
    return path


def _deepseek_v4_tensors(n_layers: int = 2, n_experts: int = 256, *, drop=None, tamper_rank=None, n_shared: int = 1) -> dict:
    """Build a deepseek-v4-shaped tensor dict (MLA + per-expert MoE)."""
    HIDDEN = 7168
    Q_LORA = 1536
    KV_LORA = 512
    KV_A_OUT = 576
    t: dict[str, tuple[str, list[int]]] = {}
    drop = drop or set()
    for li in range(n_layers):
        mla = {
            f"model.layers.{li}.self_attn.q_a_proj.weight": ("BF16", [Q_LORA, HIDDEN]),
            f"model.layers.{li}.self_attn.q_a_layernorm.weight": ("BF16", [Q_LORA]),
            f"model.layers.{li}.self_attn.q_b_proj.weight": ("BF16", [16384, Q_LORA]),
            f"model.layers.{li}.self_attn.kv_a_proj_with_mqa.weight": ("BF16", [KV_A_OUT, HIDDEN]),
            f"model.layers.{li}.self_attn.kv_a_layernorm.weight": ("BF16", [KV_LORA]),
            f"model.layers.{li}.self_attn.kv_b_proj.weight": ("BF16", [32768, KV_LORA]),
        }
        for k, v in mla.items():
            if k in drop:
                continue
            t[k] = v
        # router
        t[f"model.layers.{li}.mlp.gate.weight"] = ("BF16", [n_experts, HIDDEN])
        # per-expert projections
        for ei in range(n_experts):
            t[f"model.layers.{li}.mlp.experts.{ei}.gate_proj.weight"] = ("BF16", [HIDDEN * 2, HIDDEN])
            t[f"model.layers.{li}.mlp.experts.{ei}.down_proj.weight"] = ("BF16", [HIDDEN, HIDDEN * 2])
            t[f"model.layers.{li}.mlp.experts.{ei}.up_proj.weight"] = ("BF16", [HIDDEN * 2, HIDDEN])
        # shared expert
        if li >= n_layers - n_shared + 1 or n_shared >= 1:
            for _s in range(n_shared):
                t[f"model.layers.{li}.mlp.shared_experts.gate_proj.weight"] = ("BF16", [HIDDEN * 2, HIDDEN])
        # apply rank tamper
        if tamper_rank and li == 0:
            t[f"model.layers.{li}.self_attn.q_a_proj.weight"] = ("BF16", [tamper_rank, HIDDEN])
    # global tensors
    t["model.embed_tokens.weight"] = ("BF16", [257024, HIDDEN])
    t["model.norm.weight"] = ("BF16", [HIDDEN])
    return t


def _kimi_k3_tensors(n_layers: int = 2, n_experts: int = 896, *, wrong_experts: int | None = None) -> dict:
    """Build a kimi-k3-shaped tensor dict (batched MoE, standard GQA)."""
    HIDDEN = 6144
    t: dict[str, tuple[str, list[int]]] = {}
    ne = wrong_experts if wrong_experts is not None else n_experts
    for li in range(n_layers):
        t[f"model.layers.{li}.self_attn.q_proj.weight"] = ("BF16", [HIDDEN * 4, HIDDEN])
        t[f"model.layers.{li}.self_attn.k_proj.weight"] = ("BF16", [1024, HIDDEN])
        t[f"model.layers.{li}.self_attn.v_proj.weight"] = ("BF16", [1024, HIDDEN])
        t[f"model.layers.{li}.self_attn.o_proj.weight"] = ("BF16", [HIDDEN, HIDDEN * 4])
        t[f"model.layers.{li}.mlp.gate.weight"] = ("BF16", [ne, HIDDEN])
        # batched experts: leading dim = n_experts
        t[f"model.layers.{li}.mlp.experts.gate_proj.weight"] = ("BF16", [ne, HIDDEN * 2, HIDDEN])
        t[f"model.layers.{li}.mlp.experts.down_proj.weight"] = ("BF16", [ne, HIDDEN, HIDDEN * 2])
        t[f"model.layers.{li}.mlp.experts.up_proj.weight"] = ("BF16", [ne, HIDDEN * 2, HIDDEN])
        t[f"model.layers.{li}.mlp.shared_experts.gate_proj.weight"] = ("BF16", [HIDDEN * 2, HIDDEN])
    t["model.embed_tokens.weight"] = ("BF16", [160000, HIDDEN])
    t["model.norm.weight"] = ("BF16", [HIDDEN])
    return t


def _qwen3_7_tensors(n_layers: int = 2, n_experts: int = 128, n_shared: int = 4, *, drop_shared: bool = False) -> dict:
    """Build a qwen3.7-shaped tensor dict (batched MoE + shared experts)."""
    HIDDEN = 4096
    t: dict[str, tuple[str, list[int]]] = {}
    for li in range(n_layers):
        t[f"model.layers.{li}.self_attn.q_proj.weight"] = ("BF16", [HIDDEN * 4, HIDDEN])
        t[f"model.layers.{li}.self_attn.k_proj.weight"] = ("BF16", [1024, HIDDEN])
        t[f"model.layers.{li}.self_attn.v_proj.weight"] = ("BF16", [1024, HIDDEN])
        t[f"model.layers.{li}.self_attn.o_proj.weight"] = ("BF16", [HIDDEN, HIDDEN * 4])
        t[f"model.layers.{li}.mlp.gate.weight"] = ("BF16", [n_experts, HIDDEN])
        t[f"model.layers.{li}.mlp.experts.gate_proj.weight"] = ("BF16", [n_experts, HIDDEN * 2, HIDDEN])
        t[f"model.layers.{li}.mlp.experts.down_proj.weight"] = ("BF16", [n_experts, HIDDEN, HIDDEN * 2])
        t[f"model.layers.{li}.mlp.experts.up_proj.weight"] = ("BF16", [n_experts, HIDDEN * 2, HIDDEN])
        if not drop_shared:
            t[f"model.layers.{li}.mlp.shared_experts.gate_proj.weight"] = ("BF16", [HIDDEN * 2, HIDDEN])
    t["model.embed_tokens.weight"] = ("BF16", [151936, HIDDEN])
    t["model.norm.weight"] = ("BF16", [HIDDEN])
    return t


# --- gguf fixture writer --------------------------------------------------------

def _gguf_string(buf: bytearray, s: str) -> None:
    b = s.encode("utf-8")
    buf += struct.pack("<Q", len(b)) + b


def _write_gguf(path: str, tensors: dict, fields: dict | None = None) -> str:
    """Write a minimal valid GGUF v3 file (metadata + tensor-info table; tiny data)."""
    GGUF_MAGIC = b"GGUF"
    version = 3
    n_tensors = len(tensors)
    kv = fields or {}
    n_kv = len(kv)

    _STRING, _UINT32, _UINT64, _ARRAY, _BOOL, _F32 = 8, 4, 10, 9, 7, 6

    body = bytearray()
    # KV metadata
    for k, v in kv.items():
        _gguf_string(body, k)
        if isinstance(v, bool):
            body += struct.pack("<I", _BOOL) + struct.pack("<B", int(v))
        elif isinstance(v, int):
            body += struct.pack("<I", _UINT32) + struct.pack("<I", v)
        elif isinstance(v, str):
            body += struct.pack("<I", _STRING)
            _gguf_string(body, v)
        elif isinstance(v, list):
            body += struct.pack("<I", _ARRAY)
            body += struct.pack("<I", _UINT32)
            body += struct.pack("<Q", len(v))
            for item in v:
                body += struct.pack("<I", item)
        else:
            raise TypeError(f"unsupported gguf kv value type: {type(v)}")
    # tensor infos
    for name, (dtype, dims) in tensors.items():
        _gguf_string(body, name)
        body += struct.pack("<I", len(dims))
        for d in dims:
            body += struct.pack("<Q", int(d))
        # dtype id: F32=0, BF16=35, F16=1
        dtype_id = {"F32": 0, "BF16": 35, "F16": 1}.get(str(dtype).upper(), 0)
        body += struct.pack("<I", dtype_id)
        body += struct.pack("<Q", 0)  # offset

    header = struct.pack("<4sIQQ", GGUF_MAGIC, version, n_tensors, n_kv)
    with open(path, "wb") as f:
        f.write(header)
        f.write(body)
        f.write(b"\x00" * 64)  # tiny data region
    return path


# --- fixtures -------------------------------------------------------------------

@pytest.fixture
def tmpdir_path(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def deepseek_safetensors(tmp_path: Path) -> str:
    p = tmp_path / "deepseek-v4.safetensors"
    return _write_safetensors(str(p), _deepseek_v4_tensors())


@pytest.fixture
def deepseek_safetensors_missing_kv_proj(tmp_path: Path) -> str:
    """Drop one MLA projection from layer 0 -> incomplete MLA anomaly."""
    drop = {f"model.layers.0.self_attn.kv_b_proj.weight"}
    p = tmp_path / "deepseek-broken.safetensors"
    return _write_safetensors(str(p), _deepseek_v4_tensors(drop=drop))


@pytest.fixture
def deepseek_safetensors_wrong_rank(tmp_path: Path) -> str:
    """q_a_proj with the wrong q_lora_rank -> rank mismatch anomaly."""
    p = tmp_path / "deepseek-wrongrank.safetensors"
    return _write_safetensors(str(p), _deepseek_v4_tensors(tamper_rank=1024))


@pytest.fixture
def deepseek_safetensors_wrong_expert_count(tmp_path: Path) -> str:
    """255 experts instead of 256 -> expert count mismatch."""
    p = tmp_path / "deepseek-wrongexperts.safetensors"
    return _write_safetensors(str(p), _deepseek_v4_tensors(n_experts=255))


@pytest.fixture
def deepseek_gguf(tmp_path: Path) -> str:
    """DeepSeek-V4 GGUF — batched experts (ffn_gate_exps shape[0]=256)."""
    p = tmp_path / "deepseek-v4.gguf"
    HIDDEN = 7168
    tensors = {}
    for li in range(2):
        tensors[f"blk.{li}.attn_q_a.weight"] = ("BF16", [1536, HIDDEN])
        tensors[f"blk.{li}.attn_q_a_norm.weight"] = ("BF16", [1536])
        tensors[f"blk.{li}.attn_q_b.weight"] = ("BF16", [16384, 1536])
        tensors[f"blk.{li}.attn_kv_a_mqa.weight"] = ("BF16", [576, HIDDEN])
        tensors[f"blk.{li}.attn_kv_a_norm.weight"] = ("BF16", [512])
        tensors[f"blk.{li}.attn_kv_b.weight"] = ("BF16", [32768, 512])
        tensors[f"blk.{li}.ffn_gate_inp.weight"] = ("BF16", [256, HIDDEN])
        tensors[f"blk.{li}.ffn_gate_exps.weight"] = ("BF16", [256, HIDDEN * 2, HIDDEN])
        tensors[f"blk.{li}.ffn_down_exps.weight"] = ("BF16", [256, HIDDEN, HIDDEN * 2])
        tensors[f"blk.{li}.ffn_up_exps.weight"] = ("BF16", [256, HIDDEN * 2, HIDDEN])
        tensors[f"blk.{li}.ffn_gate_shexp.weight"] = ("BF16", [HIDDEN * 2, HIDDEN])
    tensors["token_embd.weight"] = ("BF16", [257024, HIDDEN])
    return _write_gguf(str(p), tensors, fields={"general.architecture": "deepseek", "general.name": "deepseek-v4"})


@pytest.fixture
def deepseek_gguf_wrong_experts(tmp_path: Path) -> str:
    p = tmp_path / "deepseek-v4-bad.gguf"
    HIDDEN = 7168
    tensors = {}
    for li in range(2):
        tensors[f"blk.{li}.attn_q_a.weight"] = ("BF16", [1536, HIDDEN])
        tensors[f"blk.{li}.attn_q_a_norm.weight"] = ("BF16", [1536])
        tensors[f"blk.{li}.attn_q_b.weight"] = ("BF16", [16384, 1536])
        tensors[f"blk.{li}.attn_kv_a_mqa.weight"] = ("BF16", [576, HIDDEN])
        tensors[f"blk.{li}.attn_kv_a_norm.weight"] = ("BF16", [512])
        tensors[f"blk.{li}.attn_kv_b.weight"] = ("BF16", [32768, 512])
        tensors[f"blk.{li}.ffn_gate_inp.weight"] = ("BF16", [256, HIDDEN])
        # batched expert count = 200, not 256
        tensors[f"blk.{li}.ffn_gate_exps.weight"] = ("BF16", [200, HIDDEN * 2, HIDDEN])
        tensors[f"blk.{li}.ffn_gate_shexp.weight"] = ("BF16", [HIDDEN * 2, HIDDEN])
    return _write_gguf(str(p), tensors, fields={"general.architecture": "deepseek"})


@pytest.fixture
def kimi_safetensors(tmp_path: Path) -> str:
    p = tmp_path / "kimi-k3.safetensors"
    return _write_safetensors(str(p), _kimi_k3_tensors())


@pytest.fixture
def kimi_safetensors_wrong_experts(tmp_path: Path) -> str:
    p = tmp_path / "kimi-k3-bad.safetensors"
    return _write_safetensors(str(p), _kimi_k3_tensors(wrong_experts=900))


@pytest.fixture
def qwen_safetensors(tmp_path: Path) -> str:
    p = tmp_path / "qwen3.7.safetensors"
    return _write_safetensors(str(p), _qwen3_7_tensors())


@pytest.fixture
def qwen_safetensors_no_shared(tmp_path: Path) -> str:
    p = tmp_path / "qwen3.7-noshared.safetensors"
    return _write_safetensors(str(p), _qwen3_7_tensors(drop_shared=True))


@pytest.fixture
def sharded_deepseek_dir(tmp_path: Path) -> str:
    """A multi-shard safetensors directory + index.json."""
    tensors = _deepseek_v4_tensors()
    # split layer 0 into shard 1, layer 1 into shard 2, globals into shard 1
    shard1 = {}
    shard2 = {}
    weight_map = {}
    for name, val in tensors.items():
        if "layers.0" in name or "embed_tokens" in name or name == "model.norm.weight":
            shard1[name] = val
            weight_map[name] = "model-00001-of-00002.safetensors"
        else:
            shard2[name] = val
            weight_map[name] = "model-00002-of-00002.safetensors"
    _write_safetensors(str(tmp_path / "model-00001-of-00002.safetensors"), shard1)
    _write_safetensors(str(tmp_path / "model-00002-of-00002.safetensors"), shard2)
    index = {
        "metadata": {"total_size": 1},
        "weight_map": weight_map,
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    return str(tmp_path)


@pytest.fixture
def evil_pickle(tmp_path: Path) -> str:
    """A genuinely malicious pickle (REDUCE -> os.system) for the exploit test."""
    import os as _os

    class _Bad:
        def __reduce__(self):
            return (_os.system, ("echo pwn",))

    p = tmp_path / "evil.pkl"
    with open(p, "wb") as f:
        pickle.dump(_Bad(), f)
    return str(p)


@pytest.fixture
def benign_pickle(tmp_path: Path) -> str:
    """A benign pickle (plain dict, no code-exec)."""
    p = tmp_path / "config.pkl"
    with open(p, "wb") as f:
        pickle.dump({"hidden_size": 4096, "n_layers": 2}, f)
    return str(p)


@pytest.fixture
def mixed_model_dir(tmp_path: Path, deepseek_safetensors, evil_pickle) -> str:
    """A model dir with a clean safetensors + a poisoned .pkl alongside."""
    # copy the safetensors into the dir
    import shutil
    target = tmp_path / "model.safetensors"
    shutil.copy(deepseek_safetensors, target)
    shutil.copy(evil_pickle, tmp_path / "config.bin")
    return str(tmp_path)
