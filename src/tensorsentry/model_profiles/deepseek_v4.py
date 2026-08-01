"""DeepSeek-V4 profile — Multi-head Latent Attention (MLA) + MoE.

DeepSeek-V3 introduced MLA: instead of materialising per-head Q/K/V, queries and
KV cache are projected down to a low-rank latent (``q_lora_rank``/``kv_lora_rank``)
and re-expanded. The lora ranks are the load-bearing structural invariant — an
attacker who quietly mutated them would change the attention computation
without touching any pickle byte, so generic scanners can never see it.

DeepSeek-V4 keeps the MLA block and the 256-routed-expert + 1-shared MoE. The
profile encodes both:

* MLA projections with the canonical ranks
  (``q_lora_rank=1536``, ``kv_lora_rank=512``, ``qk_rope_head_dim=64``);
* 256 per-expert projections (``gate_proj`` / ``down_proj`` / ``up_proj``) plus a
  shared expert and a router.

The validator confirms every MoE layer present in a shard carries the full MLA
set at the right ranks and exactly 256 experts — the per-model schema a
generic scanner structurally cannot encode.
"""

from __future__ import annotations

from ..tensor_validate import MLASpec, MoESpec, TensorProfile, TensorSpec

# DeepSeek-V3/V4 hidden=7168, q_lora_rank=1536, kv_lora_rank=512, n_heads=128,
# qk_nope_head_dim=128, qk_rope_head_dim=64, n_routed_experts=256, n_shared=1.
HIDDEN = 7168
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
N_HEADS = 128
QK_ROPE_HEAD_DIM = 64
KV_A_PROJ_OUT = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
N_EXPERTS = 256

_PROFILE = TensorProfile(
    model_id="deepseek-v4",
    architecture="DeepSeek-V4 MLA + MoE (256 routed + 1 shared)",
    mla=MLASpec(
        q_lora_rank=Q_LORA_RANK,
        kv_lora_rank=KV_LORA_RANK,
        n_heads=N_HEADS,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
    ),
    moe=MoESpec(
        n_experts=N_EXPERTS,
        n_shared=1,
        # HF/safetensors: per-expert tensors
        expert_pattern="model.layers.*.mlp.experts.*.gate_proj.weight",
        # GGUF: llama.cpp batches DeepSeek experts as ffn_gate_exps[256, ...]
        expert_gguf_pattern="blk.*.ffn_gate_exps.weight",
        shared_pattern="model.layers.*.mlp.shared_experts.gate_proj.weight",
        shared_gguf_pattern="blk.*.ffn_gate_shexp.weight",
        router_pattern="model.layers.*.mlp.gate.weight",
        router_gguf_pattern="blk.*.ffn_gate_inp.weight",
    ),
    required_tensors=(
        # --- MLA projections (the headline structural invariant) ---
        TensorSpec(
            logical_name="q_a_proj",
            pattern="model.layers.*.self_attn.q_a_proj.weight",
            gguf_pattern="blk.*.attn_q_a.weight",
            dtype="BF16",
            shape_constraints={"ndim": 2, "out_rank": Q_LORA_RANK},
            category="mla",
        ),
        TensorSpec(
            logical_name="q_a_layernorm",
            pattern="model.layers.*.self_attn.q_a_layernorm.weight",
            gguf_pattern="blk.*.attn_q_a_norm.weight",
            dtype="BF16",
            shape_constraints={"ndim": 1, "dim0_eq": Q_LORA_RANK},
            category="mla",
        ),
        TensorSpec(
            logical_name="q_b_proj",
            pattern="model.layers.*.self_attn.q_b_proj.weight",
            gguf_pattern="blk.*.attn_q_b.weight",
            dtype="BF16",
            shape_constraints={"ndim": 2, "in_rank": Q_LORA_RANK},
            category="mla",
        ),
        TensorSpec(
            logical_name="kv_a_proj_with_mqa",
            pattern="model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
            gguf_pattern="blk.*.attn_kv_a_mqa.weight",
            dtype="BF16",
            shape_constraints={"ndim": 2, "out_rank": KV_A_PROJ_OUT},
            category="mla",
        ),
        TensorSpec(
            logical_name="kv_a_layernorm",
            pattern="model.layers.*.self_attn.kv_a_layernorm.weight",
            gguf_pattern="blk.*.attn_kv_a_norm.weight",
            dtype="BF16",
            shape_constraints={"ndim": 1, "dim0_eq": KV_LORA_RANK},
            category="mla",
        ),
        TensorSpec(
            logical_name="kv_b_proj",
            pattern="model.layers.*.self_attn.kv_b_proj.weight",
            gguf_pattern="blk.*.attn_kv_b.weight",
            dtype="BF16",
            shape_constraints={"ndim": 2, "in_rank": KV_LORA_RANK},
            category="mla",
        ),
        # --- MoE router (shared expert + router existence checked by moe spec) ---
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
