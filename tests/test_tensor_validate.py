"""m1 tests — safetensors reader + tensor_validate + deepseek-v4 MLA profile.

These exercise the headline MLA structural invariant on real .safetensors bytes
(header-only): a clean DeepSeek-V4 shard is OK, a tampered one (missing MLA
projection, wrong lora rank, wrong expert count) is flagged.
"""

from __future__ import annotations

import os

import pytest

from tensorsentry import safetensors_reader, scanner, tensor_validate
from tensorsentry.model_profiles import get_profile, list_profiles
from tensorsentry.tensor_validate import (
    Anomaly,
    StructureResult,
    TensorView,
    check_dtype,
    check_shape,
    validate_structure,
)


# --- profile registry ----------------------------------------------------------

def test_three_profiles_registered():
    ids = {p.model_id for p in list_profiles()}
    assert ids == {"deepseek-v4", "kimi-k3", "qwen3.7"}


def test_deepseek_v4_profile_has_mla_and_moe():
    p = get_profile("deepseek-v4")
    assert p is not None
    assert p.mla is not None
    assert p.mla.q_lora_rank == 1536
    assert p.mla.kv_lora_rank == 512
    assert p.moe is not None
    assert p.moe.n_experts == 256
    # every MLA spec is flagged category="mla"
    mla_specs = [s for s in p.required_tensors if s.category == "mla"]
    assert {s.logical_name for s in mla_specs} == {
        "q_a_proj", "q_a_layernorm", "q_b_proj",
        "kv_a_proj_with_mqa", "kv_a_layernorm", "kv_b_proj",
    }


# --- safetensors reader --------------------------------------------------------

def test_safetensors_reader_parses_header(deepseek_safetensors):
    h = safetensors_reader.read_header(deepseek_safetensors)
    # MLA projection present with the canonical rank
    assert "model.layers.0.self_attn.q_a_proj.weight" in h.tensors
    q_a = h.tensors["model.layers.0.self_attn.q_a_proj.weight"]
    assert q_a.dtype == "BF16"
    assert q_a.shape[0] == 1536  # q_lora_rank
    assert q_a.ndim == 2
    # header-only parse: data_offset points past the JSON header (no weights read)
    assert h.data_offset > 8
    # metadata absent in our fixture -> empty dict
    assert isinstance(h.metadata, dict)


def test_safetensors_reader_rejects_non_safetensors(tmp_path):
    bad = tmp_path / "not.st"
    bad.write_bytes(b"not a safetensors file at all, just text")
    assert not safetensors_reader.is_safetensors(str(bad))
    with pytest.raises(safetensors_reader.SafetensorsError):
        safetensors_reader.read_header(str(bad))


def test_safetensors_directory_and_shards(sharded_deepseek_dir):
    headers = safetensors_reader.read_directory(sharded_deepseek_dir)
    assert len(headers) == 2
    merged = safetensors_reader.merge(headers)
    # the merged header carries tensors from both shards
    assert "model.layers.0.self_attn.q_a_proj.weight" in merged.tensors
    assert "model.layers.1.self_attn.q_a_proj.weight" in merged.tensors


# --- tensor_validate primitives ------------------------------------------------

def test_check_shape_rank_mismatch():
    spec = tensor_validate.TensorSpec(logical_name="x", pattern="*", shape_constraints={"out_rank": 1536})

    class T:
        name = "a"
        dtype = "BF16"
        shape = (1024, 7168)
        ndim = 2

    assert check_shape(spec, T) is not None  # mismatch reported


def test_check_dtype_passes_on_match():
    spec = tensor_validate.TensorSpec(logical_name="x", pattern="*", dtype="BF16")

    class T:
        name = "a"
        dtype = "bf16"
        shape = (1,)
        ndim = 1

    assert check_dtype(spec, T) is None  # case-insensitive match


# --- m1 end-to-end: validate against real .safetensors -------------------------

def test_validate_deepseek_v4_clean(deepseek_safetensors):
    report = scanner.validate("deepseek-v4", deepseek_safetensors)
    assert report.structure == "ok", report.anomalies
    assert report.exploit == "clean"
    assert report.model == "deepseek-v4"
    assert report.source_format == "safetensors"
    assert report.structure_detail["expert_count"] == 256
    assert report.ok


def test_validate_flags_missing_mla_projection(deepseek_safetensors_missing_kv_proj):
    report = scanner.validate("deepseek-v4", deepseek_safetensors_missing_kv_proj)
    assert report.structure == "anomaly"
    assert any("incomplete_mla" in a for a in report.anomalies), report.anomalies


def test_validate_flags_wrong_lora_rank(deepseek_safetensors_wrong_rank):
    report = scanner.validate("deepseek-v4", deepseek_safetensors_wrong_rank)
    assert report.structure == "anomaly"
    assert any("malformed_tensor" in a and "q_a_proj" in a for a in report.anomalies)


def test_validate_flags_wrong_expert_count(deepseek_safetensors_wrong_expert_count):
    report = scanner.validate("deepseek-v4", deepseek_safetensors_wrong_expert_count)
    assert report.structure == "anomaly"
    assert any("expert_count_mismatch" in a for a in report.anomalies), report.anomalies


def test_validate_unknown_profile_returns_unknown(tmp_path, deepseek_safetensors):
    report = scanner.validate("no-such-model", deepseek_safetensors)
    assert report.structure == "unknown_profile"


def test_validate_sharded_directory_ok(sharded_deepseek_dir):
    report = scanner.validate("deepseek-v4", sharded_deepseek_dir)
    # Both shards together cover layers 0+1 with full MLA — OK.
    assert report.structure == "ok", report.anomalies


def test_validate_gguf_clean(deepseek_gguf):
    report = scanner.validate("deepseek-v4", deepseek_gguf)
    assert report.source_format == "gguf"
    assert report.structure == "ok", report.anomalies
    assert report.structure_detail["expert_count"] == 256


def test_validate_gguf_wrong_expert_count(deepseek_gguf_wrong_experts):
    report = scanner.validate("deepseek-v4", deepseek_gguf_wrong_experts)
    assert report.structure == "anomaly"
    assert any("expert_count_mismatch" in a for a in report.anomalies), report.anomalies
