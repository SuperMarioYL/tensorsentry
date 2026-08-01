"""Qwen3.7 profile — MoE with shared experts (Alibaba).

Qwen3 MoE ships a moderate expert count (~128 routed + 4 shared) with the
shared-expert pattern Qwen introduced. TensorSentry encodes the expert count
and the shared-expert presence as the structural invariant — a quiet attacker
who dropped a shared expert would silently change routing without tripping any
pickle scanner.
"""

from __future__ import annotations

from ..tensor_validate import MoESpec, TensorProfile, TensorSpec

HIDDEN = 4096
N_EXPERTS = 128
N_SHARED = 4

_PROFILE = TensorProfile(
    model_id="qwen3.7",
    architecture="Qwen3.7 MoE (128 routed + 4 shared)",
    mla=None,
    moe=MoESpec(
        n_experts=N_EXPERTS,
        n_shared=N_SHARED,
        # Qwen batches experts per projection with leading expert axis = 128.
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
        TensorSpec(
            logical_name="shared_expert",
            pattern="model.layers.*.mlp.shared_experts.gate_proj.weight",
            gguf_pattern="blk.*.ffn_gate_shexp.weight",
            dtype="BF16",
            shape_constraints={"ndim": 2, "in_rank": HIDDEN},
            category="moe",
            min_matches=0,  # present-only check; not every shard has shared experts
        ),
    ),
)

PROFILE = _PROFILE
