"""m2 tests — gguf reader, pickle exploit detection, combined scan, CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tensorsentry import gguf_reader, pickle_scan, report, scanner
from tensorsentry.cli import main as cli_main


# --- gguf reader ---------------------------------------------------------------

def test_gguf_reader_parses_header(deepseek_gguf):
    h = gguf_reader.read_header(deepseek_gguf)
    assert h.version == 3
    assert h.fields.get("general.architecture") == "deepseek"
    # MLA + batched experts present
    assert "blk.0.attn_q_a.weight" in h.tensors
    assert h.tensors["blk.0.ffn_gate_exps.weight"].shape[0] == 256
    # tensor dtype decoded
    assert h.tensors["blk.0.attn_q_a.weight"].dtype == "BF16"


def test_gguf_reader_rejects_non_gguf(tmp_path):
    bad = tmp_path / "not.bin"
    bad.write_bytes(b"NOPE" + b"\x00" * 32)
    assert not gguf_reader.is_gguf(str(bad))
    with pytest.raises(gguf_reader.GGUFError):
        gguf_reader.read_header(str(bad))


# --- pickle scan ---------------------------------------------------------------

def test_pickle_scan_safetensors_is_clean(deepseek_safetensors):
    """A .safetensors file is not pickle -> clean verdict."""
    res = pickle_scan.scan_path(deepseek_safetensors)
    assert res.exploit == "clean"
    assert "no __reduce__" in res.reason or res.scanned_files == 0


def test_pickle_scan_flags_malicious_pickle(evil_pickle):
    res = pickle_scan.scan_path(evil_pickle)
    assert res.exploit == "suspect"
    quals = {f.qualname for f in res.findings if f.safety == "dangerous"}
    assert any("system" in q for q in quals), res.findings
    assert res.infected_files >= 1


def test_pickle_scan_benign_pickle_is_pickle_found(benign_pickle):
    res = pickle_scan.scan_path(benign_pickle)
    # benign pickle: no dangerous globals -> pickle_found (informational) or clean
    assert res.exploit in ("pickle_found", "clean")
    assert not any(f.safety == "dangerous" for f in res.findings)


def test_pickle_scan_directory_with_poison(mixed_model_dir):
    res = pickle_scan.scan_path(mixed_model_dir)
    assert res.exploit == "suspect"
    assert res.scanned_files >= 1


# --- combined scan (m2) --------------------------------------------------------

def test_scan_clean_safetensors(deepseek_safetensors):
    rep = scanner.scan(deepseek_safetensors, model_id="deepseek-v4")
    assert rep.structure == "ok", rep.anomalies
    assert rep.exploit == "clean"
    assert rep.provenance == "unreachable"  # m3 stub
    assert rep.ok


def test_scan_clean_gguf(deepseek_gguf):
    rep = scanner.scan(deepseek_gguf, model_id="deepseek-v4")
    assert rep.source_format == "gguf"
    assert rep.structure == "ok"
    assert rep.exploit == "clean"
    assert rep.ok


def test_scan_flags_anomaly_and_still_clean_exploit(deepseek_safetensors_wrong_rank):
    rep = scanner.scan(deepseek_safetensors_wrong_rank, model_id="deepseek-v4")
    assert rep.structure == "anomaly"
    assert rep.exploit == "clean"  # not pickle
    assert not rep.ok


def test_scan_kimi_expert_count(kimi_safetensors):
    rep = scanner.scan(kimi_safetensors, model_id="kimi-k3")
    assert rep.structure == "ok", rep.anomalies
    assert rep.structure_detail["expert_count"] == 896


def test_scan_kimi_wrong_expert_count(kimi_safetensors_wrong_experts):
    rep = scanner.scan(kimi_safetensors_wrong_experts, model_id="kimi-k3")
    assert rep.structure == "anomaly"
    assert any("expert_count_mismatch" in a for a in rep.anomalies)


def test_scan_qwen_clean(qwen_safetensors):
    rep = scanner.scan(qwen_safetensors, model_id="qwen3.7")
    assert rep.structure == "ok", rep.anomalies
    assert rep.structure_detail["expert_count"] == 128


def test_scan_no_model_runs_exploit_only(mixed_model_dir):
    rep = scanner.scan(mixed_model_dir, model_id=None)
    assert rep.exploit == "suspect"
    assert rep.structure == "unknown_profile"  # no profile requested


def test_scan_provenance_stub_raises(mixed_model_dir):
    from tensorsentry.provenance import ProvenanceNotImplemented, verify_provenance
    with pytest.raises(ProvenanceNotImplemented):
        verify_provenance(mixed_model_dir, "deepseek-v4")


# --- report JSON / rich --------------------------------------------------------

def test_report_json_roundtrip(deepseek_safetensors):
    rep = scanner.scan(deepseek_safetensors, model_id="deepseek-v4")
    js = report.to_json(rep)
    obj = json.loads(js)
    assert obj["model"] == "deepseek-v4"
    assert obj["structure"] == "ok"
    assert obj["exploit"] == "clean"


def test_report_rich_renders(deepseek_safetensors):
    rep = scanner.scan(deepseek_safetensors, model_id="deepseek-v4")
    rendered = report.render_rich(rep)
    # rich renderables are composable objects; just assert it builds.
    assert rendered is not None


# --- CLI -----------------------------------------------------------------------

def _run_cli(*args) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    return subprocess.run(
        [sys.executable, "-m", "tensorsentry.cli", *args],
        capture_output=True, text=True, env=env,
    )


def test_cli_profiles():
    proc = _run_cli("profiles")
    assert proc.returncode == 0
    assert "deepseek-v4" in proc.stdout
    assert "kimi-k3" in proc.stdout
    assert "qwen3.7" in proc.stdout


def test_cli_validate_clean(deepseek_safetensors):
    proc = _run_cli("validate", "deepseek-v4", deepseek_safetensors)
    assert proc.returncode == 0, proc.stderr
    assert "STRUCTURE" in proc.stdout


def test_cli_validate_anomaly_exits_nonzero(deepseek_safetensors_wrong_rank):
    proc = _run_cli("validate", "deepseek-v4", deepseek_safetensors_wrong_rank)
    assert proc.returncode == 1
    assert "anomaly" in proc.stdout.lower()


def test_cli_scan_json(deepseek_safetensors):
    proc = _run_cli("scan", "--model", "deepseek-v4", "--json", deepseek_safetensors)
    assert proc.returncode == 0, proc.stderr
    obj = json.loads(proc.stdout)
    assert obj["structure"] == "ok"
    assert obj["model"] == "deepseek-v4"


def test_cli_scan_suspect_pickle_exits_nonzero(mixed_model_dir):
    proc = _run_cli("scan", "--model", "deepseek-v4", "--quiet", mixed_model_dir)
    assert proc.returncode == 1
    assert "suspect" in proc.stdout


def test_cli_version():
    proc = _run_cli("--version")
    assert proc.returncode == 0
    assert "0.1.0" in proc.stdout
