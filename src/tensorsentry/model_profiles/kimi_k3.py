"""Kimi K3 profile — 896-expert MoE (Moonshot).

Kimi K3's structural headline is the very large expert count (~896 routed
experts), an order of magnitude beyond DeepSeek-V4's 256. Generic scanners
cannot assert "this weight file exposes exactly 896 Kimi experts" because that
requires a maintained per-model schema — exactly what TensorSentry encodes.

This profile focuses on the MoE expert count and standard grouped-query
attention projections. (Kimi's attention block specifics are tracked as a
profile refinement; the m1/m2 wedge validates the expert-count invariant the
plan names.)
"""

from __future__ import annotations

from ..tensor_validate import MoESpec, TensorProfile, TensorSpec

HIDDEN = 6144
N_EXPERTS = 896

_PROFILE = TensorProfile(
    model_id="kimi-k3",
    architecture="Kimi K3 MoE (896 routed experts)",
    mla=None,
    moe=MoESpec(
        n_experts=N_EXPERTS,
        n_shared=1,
        # Kimi batches experts: one tensor per (layer, projection) with leading
        # expert axis = 896. The validator reads shape[0].
        expert_pattern="model.layers.*.mlp.experts.gate_proj.weight",
        expert_gguf_pattern="blk.*.ffn_gate_exps.weight",
        shared_pattern="model.layers.*.mlp.shared_experts.gate_proj.weight",
        shared_gguf_pattern="blk.*.ffn_gate_shexp.weight",
        router_pattern="model.layers.*.mlp.gate.weight",
        router_gguf_pattern="blk.*.ffn_gate_inp.weight",
    ),
    required_tensors=(
        TensorSpec(
            logical_name="q_proj",
            pattern="model.layers.*.self_attn.q_proj.weight",
            gguf_pattern="blk.*.attn_q.weight",
            dtype="BF16",
            shape_constraints={"ndim": 2, "in_rank": HIDDEN},
            category="attention",
        ),
        TensorSpec(
            logical_name="k_proj",
            pattern="model.layers.*.self_attn.k_proj.weight",
            gguf_pattern="blk.*.attn_k.weight",
            dtype="BF16",
            shape_constraints={"ndim": 2, "in_rank": HIDDEN},
            category="attention",
        ),
        TensorSpec(
            logical_name="v_proj",
            pattern="model.layers.*.self_attn.v_proj.weight",
            gguf_pattern="blk.*.attn_v.weight",
            dtype="BF16",
            shape_constraints={"ndim": 2, "in_rank": HIDDEN},
            category="attention",
        ),
        TensorSpec(
            logical_name="o_proj",
            pattern="model.layers.*.self_attn.o_proj.weight",
            gguf_pattern="blk.*.attn_output.weight",
            dtype="BF16",
            shape_constraints={"ndim": 2, "out_rank": HIDDEN},
            category="attention",
        ),
        TensorSpec(
            logical_name="router",
            pattern="model.layers.*.mlp.gate.weight",
            gguf_pattern="blk.*.ffn_gate_inp.weight",
            dtype="BF16",
            shape_constraints={"ndim": 2, "in_rank": HIDDEN},
            category="moe",
        ),
    ),
)

PROFILE = _PROFILE
