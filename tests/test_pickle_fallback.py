"""m5 tests — the built-in pickle-scanning fallback (no picklescan installed).

The fallback is the documented degraded path ("TensorSentry always produces an
exploit verdict"). These tests force it by blocking the picklescan import and
pin its behavior: protocol-2+ (STACK_GLOBAL) code-exec gadgets must be flagged,
distinct globals from one module must not be deduped away, non-pickle files
must be scanned without reading their whole body, and the directory extension
list must stay aligned with the primary path.
"""

from __future__ import annotations

import builtins
import os
import pickle
import sys

import pytest

from tensorsentry import pickle_scan


@pytest.fixture
def no_picklescan(monkeypatch):
    """Force scan_path onto the built-in fallback for the duration of a test."""
    # A None entry in sys.modules makes `import picklescan` raise ImportError.
    monkeypatch.setitem(sys.modules, "picklescan", None)


class _SystemCall:
    def __init__(self, func, arg):
        self._func, self._arg = func, arg

    def __reduce__(self):
        return (self._func, (self._arg,))


def _evil_two_globals_pickle(path):
    """One pickle referencing two different dangerous globals from os."""
    import os as _os

    payload = (_SystemCall(_os.system, "echo a"), _SystemCall(_os.popen, "echo b"))
    with open(path, "wb") as f:
        f.write(pickle.dumps(payload, protocol=pickle.DEFAULT_PROTOCOL))
    return path


def test_fallback_flags_protocol4_pickle(no_picklescan, tmp_path):
    """STACK_GLOBAL (protocol 2+) globals must resolve to real module/name —
    the v0.1.0 fallback recorded ('', '') and verdicted such pickles
    'pickle_found' instead of 'suspect'."""
    import os as _os

    p = tmp_path / "evil.pkl"
    with open(p, "wb") as f:
        f.write(pickle.dumps(_SystemCall(_os.system, "echo pwn")))
    res = pickle_scan.scan_path(str(p))
    assert res.tool == "builtin"
    assert res.exploit == "suspect", (res.exploit, res.reason)
    dangerous = {f.qualname for f in res.findings if f.safety == "dangerous"}
    assert any("system" in q for q in dangerous), res.findings


def test_fallback_keeps_distinct_globals_from_one_module(no_picklescan, tmp_path):
    """os.system and os.popen are two findings, not one — the v0.1.0 dedup key
    was a literal-brace bug that collapsed same-module globals."""
    p = _evil_two_globals_pickle(str(tmp_path / "evil2.pkl"))
    res = pickle_scan.scan_path(p)
    assert res.exploit == "suspect"
    dangerous = {f.qualname for f in res.findings if f.safety == "dangerous"}
    assert any("system" in q for q in dangerous), dangerous
    assert any("popen" in q for q in dangerous), dangerous


def test_fallback_benign_pickle_is_pickle_found(no_picklescan, tmp_path):
    p = tmp_path / "config.pkl"
    with open(p, "wb") as f:
        f.write(pickle.dumps({"hidden_size": 4096, "n_layers": 2}))
    res = pickle_scan.scan_path(str(p))
    assert res.exploit == "pickle_found"
    assert not any(f.safety == "dangerous" for f in res.findings)


def test_fallback_does_not_read_whole_non_pickle_file(no_picklescan, monkeypatch, tmp_path):
    """A non-pickle file is verdicted from its first bytes alone — the v0.1.0
    fallback materialized the entire file in memory first."""
    p = tmp_path / "big-weight.bin"
    with open(p, "wb") as f:
        f.write(b"\x11" * 50_000_000)  # 50 MB, not pickle magic

    read_total = {"bytes": 0}
    real_open = builtins.open

    class _Counting:
        def __init__(self, f):
            self._f = f

        def read(self, n=-1):
            data = self._f.read(n)
            read_total["bytes"] += len(data)
            return data

        def seek(self, *a):
            return self._f.seek(*a)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return self._f.__exit__(*a)

    def counting_open(path, *a, **k):
        f = real_open(path, *a, **k)
        return _Counting(f) if os.fspath(path) == os.fspath(p) else f

    monkeypatch.setattr(builtins, "open", counting_open)
    res = pickle_scan._builtin_scan_file(str(p))
    assert res.exploit == "clean"
    assert res.scanned_files == 0
    assert read_total["bytes"] < 1024, read_total["bytes"]


def test_fallback_directory_extension_parity(no_picklescan, tmp_path):
    """.dat/.data/.joblib/.ckpt pickles are scanned by the primary path, so
    the fallback must scan them too (v0.1.0 silently skipped them)."""
    import os as _os

    for ext in (".ckpt", ".dat", ".joblib", ".data"):
        p = tmp_path / f"poisoned{ext}"
        with open(p, "wb") as f:
            f.write(pickle.dumps(_SystemCall(_os.system, "echo pwn")))
    res = pickle_scan.scan_path(str(tmp_path))
    assert res.exploit == "suspect", (res.exploit, res.reason)
    assert res.scanned_files >= 4


def test_fallback_directory_reports_clean_without_pickles(no_picklescan, tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"\x00" * 64)
    (tmp_path / "notes.txt").write_text("nothing to see")
    res = pickle_scan.scan_path(str(tmp_path))
    assert res.exploit == "clean"


def test_primary_picklescan_path_still_used(deepseek_safetensors):
    """With picklescan importable the primary path runs (regression guard for
    the fallback rewrite touching shared code paths)."""
    res = pickle_scan.scan_path(deepseek_safetensors)
    assert res.tool == "picklescan"
    assert res.exploit == "clean"


def test_evil_pickle_via_primary_path(evil_pickle):
    res = pickle_scan.scan_path(evil_pickle)
    assert res.exploit == "suspect"
    assert res.tool in ("picklescan", "builtin")
    assert any("system" in f.qualname for f in res.findings if f.safety == "dangerous")
