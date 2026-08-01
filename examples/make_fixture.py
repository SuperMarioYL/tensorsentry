"""Generate a tiny but structurally-faithful DeepSeek-V4 safetensors fixture.

Usage:
    python examples/make_fixture.py [output_path]

This doubles as a programmatic-API example — it shows how TensorSentry's
readers/profiles think about a checkpoint. The fixture is header-only (a 1-byte
data region) so it is tiny yet carries the real MLA + 256-expert MoE tensor
names/shapes a ``tensorsentry validate deepseek-v4`` scan checks against.

Run the scan on the produced file:

    tensorsentry scan --model deepseek-v4 ./model.safetensors
"""

from __future__ import annotations

import json
import os
import struct
import sys


def build(n_layers: int = 2, n_experts: int = 256) -> dict:
    HIDDEN = 7168
    Q_LORA = 1536
    KV_LORA = 512
    KV_A_OUT = 576
    t: dict[str, tuple[str, list[int]]] = {}
    for li in range(n_layers):
        t[f"model.layers.{li}.self_attn.q_a_proj.weight"] = ("BF16", [Q_LORA, HIDDEN])
        t[f"model.layers.{li}.self_attn.q_a_layernorm.weight"] = ("BF16", [Q_LORA])
        t[f"model.layers.{li}.self_attn.q_b_proj.weight"] = ("BF16", [16384, Q_LORA])
        t[f"model.layers.{li}.self_attn.kv_a_proj_with_mqa.weight"] = ("BF16", [KV_A_OUT, HIDDEN])
        t[f"model.layers.{li}.self_attn.kv_a_layernorm.weight"] = ("BF16", [KV_LORA])
        t[f"model.layers.{li}.self_attn.kv_b_proj.weight"] = ("BF16", [32768, KV_LORA])
        t[f"model.layers.{li}.mlp.gate.weight"] = ("BF16", [n_experts, HIDDEN])
        for ei in range(n_experts):
            t[f"model.layers.{li}.mlp.experts.{ei}.gate_proj.weight"] = ("BF16", [HIDDEN * 2, HIDDEN])
            t[f"model.layers.{li}.mlp.experts.{ei}.down_proj.weight"] = ("BF16", [HIDDEN, HIDDEN * 2])
            t[f"model.layers.{li}.mlp.experts.{ei}.up_proj.weight"] = ("BF16", [HIDDEN * 2, HIDDEN])
        t[f"model.layers.{li}.mlp.shared_experts.gate_proj.weight"] = ("BF16", [HIDDEN * 2, HIDDEN])
    t["model.embed_tokens.weight"] = ("BF16", [257024, HIDDEN])
    t["model.norm.weight"] = ("BF16", [HIDDEN])
    return t


def write_safetensors(path: str, tensors: dict) -> str:
    header = {}
    for name, (dtype, shape) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [0, 1]}
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(raw) % 8)) % 8
    raw_padded = raw + b" " * pad
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw_padded)))
        f.write(raw_padded)
        f.write(b"\x00")
    return path


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "./model.safetensors"
    write_safetensors(out, build(n_layers=2, n_experts=256))
    print(f"wrote DeepSeek-V4-shaped fixture -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
