"""Pickle / code-exec exploit detection.

Wraps the ``picklescan`` library (do not reinvent) and falls back to a built-in
raw-byte pickle opcode scanner when the library is not installed, so TensorSentry
always produces an exploit verdict. The fallback disassembles pickle bytes with
the stdlib ``pickletools`` and matches ``GLOBAL``/``STACK_GLOBAL`` opcodes
against a curated set of dangerous modules — the same technique picklescan uses,
not a reimplementation of its scan engine.

A ``.safetensors`` / ``.gguf`` file is *not* pickle, so scanning one yields a
``clean`` verdict ("no ``__reduce__`` / code-exec tensors found"). The scanner
fires on real pickle surfaces — ``.bin``/``.pt``/``.pth``/``.pkl`` files inside a
model directory, or a malicious file mislabelled as a weight.
"""

from __future__ import annotations

import os
import pickletools
from dataclasses import dataclass, field
from typing import Any, BinaryIO

__all__ = [
    "PickleFinding",
    "PickleResult",
    "scan_path",
    "DANGEROUS_MODULES",
]

# Modules whose presence in a pickle GLOBAL is a code-exec gadget. Conservative:
# match on the module prefix so ``posix.system`` and ``os.system`` both flag.
DANGEROUS_MODULES = {
    "os", "posix", "nt", "subprocess", "builtins.eval", "builtins.exec",
    "builtins.__import__", "builtins.compile", "code", "codeop", "pdb",
    "pty", "commands", "multiprocessing", "shutil",
}
# Suspicious-but-not-auto-doom: shell helpers, file ops. Flag but lower severity.
SUSPICIOUS_MODULES = {
    "pathlib", "glob", "io.open", "webbrowser", "socket", "urllib",
    "requests", "http", "ftplib", "smtplib", "telnetlib",
    # A STACK_GLOBAL whose operand could not be resolved from the opcode
    # stream (stack-computed module/name) — treat as evasion, not innocence.
    "<unknown>",
}


@dataclass(frozen=True)
class PickleFinding:
    """One global reference found in a pickle's opcode stream."""

    module: str
    name: str
    safety: str  # "dangerous" | "suspicious" | "innocuous"

    @property
    def qualname(self) -> str:
        return f"{self.module}.{self.name}" if self.module else self.name


@dataclass
class PickleResult:
    """Result of the exploit-detection pass."""

    exploit: str  # "clean" | "suspect" | "pickle_found"
    scanned_files: int = 0
    infected_files: int = 0
    findings: list[PickleFinding] = field(default_factory=list)
    reason: str = ""
    tool: str = "skipped"  # "picklescan" | "builtin" | "skipped"

    @property
    def suspect(self) -> bool:
        return self.exploit == "suspect"


# --- pickle magic detection (used by both paths) --------------------------------

_PICKLE_MAGICS = (
    b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05", b"\x80\x06",
    b"\x80\x07", b"\x80\x08",  # protocols 2-8
    b"(lp", b"(dp", b"(S",  # legacy py2 frames
)


def _looks_like_pickle(data: bytes) -> bool:
    if not data:
        return False
    if data[:2] in _PICKLE_MAGICS or data[:1] in (b"\x80",):
        return True
    # zip / 7z containers picklescan also unpacks — leave detection to picklescan.
    return data[:2] == b"PK" or data[:6] == b"7z\xbc\xaf\x27\x1c"


def _classify(module: str, name: str) -> str:
    qual = f"{module}.{name}" if module else name
    for d in DANGEROUS_MODULES:
        if module == d or module.startswith(d + ".") or qual.startswith(d + "."):
            return "dangerous"
    for s in SUSPICIOUS_MODULES:
        if module == s or module.startswith(s + ".") or qual.startswith(s + "."):
            return "suspicious"
    return "innocuous"


# --- built-in fallback (no picklescan installed) ------------------------------

# Opcode families used to resolve STACK_GLOBAL operands (same technique as
# picklescan's _list_globals: walk back through string pushes and memo reads,
# skipping MEMOIZE/PUT markers).
_STRING_OPS = frozenset({
    "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
    "STRING", "BINSTRING", "SHORT_BINSTRING",
})
_MEMO_PUT_OPS = frozenset({"MEMOIZE", "PUT", "BINPUT", "LONG_BINPUT"})
_MEMO_GET_OPS = frozenset({"GET", "BINGET", "LONG_BINGET"})
_UNKNOWN_MODULE = "<unknown>"

# Flat pickle/pytorch file suffixes — kept aligned with the extensions the
# picklescan library scans in a directory (it additionally unpacks zip/7z/npy
# containers, which this fallback does not).
_PICKLE_SUFFIXES = (
    ".bin", ".pt", ".pth", ".pkl", ".pickle", ".dat", ".data", ".joblib", ".ckpt",
)


def _stack_global_operands(ops: list, n: int, memo: dict) -> tuple[str, str]:
    """Resolve the ``(module, name)`` operands of the STACK_GLOBAL at ``ops[n]``.

    Protocol 2+ pickles push module then name as strings (sometimes memoized)
    right before STACK_GLOBAL; walk backwards to recover them. A non-string,
    non-memo opcode means the operand was computed on the stack — an evasion
    attempt — and is reported as an unknown module rather than trusted.
    """
    values: list[str] = []
    for offset in range(1, n + 1):
        op, arg, _pos = ops[n - offset]
        name = op.name
        if name in _MEMO_PUT_OPS:
            continue
        if name in _MEMO_GET_OPS:
            try:
                key = int(arg)  # type: ignore[call-overload]
            except (TypeError, ValueError):
                values.append(_UNKNOWN_MODULE)
            else:
                val = memo.get(key, _UNKNOWN_MODULE)
                values.append(val if isinstance(val, str) else _UNKNOWN_MODULE)
        elif name in _STRING_OPS:
            values.append(str(arg))
        else:
            values.append(_UNKNOWN_MODULE)
        if len(values) == 2:
            break
    if len(values) < 2:
        return _UNKNOWN_MODULE, _UNKNOWN_MODULE
    return values[1], values[0]  # stack order: module pushed first, then name


def _disassemble_findings(stream: BinaryIO) -> list[PickleFinding]:
    """Disassemble a pickle stream with stdlib pickletools and collect globals.

    Streams the file (no full in-memory copy); a stream that stops parsing
    mid-way keeps whatever globals were found before the error, mirroring
    picklescan's partial-pickle handling.
    """
    findings: list[PickleFinding] = []
    seen: set[tuple[str, str]] = set()

    def _record(module: str, name: str) -> None:
        if (module, name) in seen:
            return
        seen.add((module, name))
        findings.append(PickleFinding(module=module, name=name, safety=_classify(module, name)))

    ops: list[tuple[Any, Any, int]] = []
    memo: dict[Any, Any] = {}
    try:
        for op, arg, pos in pickletools.genops(stream):
            ops.append((op, arg, pos))
            if op.name == "MEMOIZE":
                # MEMOIZE stores the value pushed by the *preceding* opcode.
                memo[len(memo)] = ops[-2][1] if len(ops) >= 2 else None
            elif op.name in ("PUT", "BINPUT", "LONG_BINPUT"):
                memo[arg] = ops[-2][1] if len(ops) >= 2 else None
            elif op.name in ("GLOBAL", "INST"):
                parts = str(arg or "").split(" ", 1)
                module, name = (parts[0], parts[1]) if len(parts) == 2 else (str(arg or ""), "")
                _record(module, name)
    except Exception:
        # not a valid pickle stream — keep the globals found so far
        pass

    # STACK_GLOBAL needs the full opcode list (memo may be referenced anywhere
    # earlier in the stream), so resolve it in a second pass.
    for n, (op, _arg, _pos) in enumerate(ops):
        if op.name == "STACK_GLOBAL":
            module, name = _stack_global_operands(ops, n, memo)
            _record(module, name)
    return findings


def _builtin_scan_file(path: str) -> PickleResult:
    try:
        with open(path, "rb") as f:
            head = f.read(8)
            if not _looks_like_pickle(head):
                # Not a pickle by magic — verdict without reading the (possibly
                # huge) remainder of the file.
                return PickleResult(
                    exploit="clean", tool="builtin", scanned_files=0,
                    reason="no pickle magic / __reduce__ found in file",
                )
            # Pickle magic present — rewind and disassemble the whole stream.
            f.seek(0)
            findings = _disassemble_findings(f)
    except OSError:
        return PickleResult(exploit="clean", tool="builtin", reason=f"cannot read {path}")
    dangerous = [f for f in findings if f.safety == "dangerous"]
    suspicious = [f for f in findings if f.safety == "suspicious"]
    if dangerous:
        names = ", ".join(sorted({f.qualname for f in dangerous}))
        return PickleResult(
            exploit="suspect", tool="builtin", scanned_files=1, infected_files=1,
            findings=findings,
            reason=f"dangerous code-exec globals: {names}",
        )
    if suspicious:
        names = ", ".join(sorted({f.qualname for f in suspicious}))
        return PickleResult(
            exploit="suspect", tool="builtin", scanned_files=1, infected_files=0,
            findings=findings,
            reason=f"suspicious globals: {names}",
        )
    return PickleResult(
        exploit="pickle_found", tool="builtin", scanned_files=1, infected_files=0,
        findings=findings,
        reason="pickle present, only innocuous globals",
    )


def _builtin_scan(path: str) -> PickleResult:
    if os.path.isdir(path):
        results: list[PickleResult] = []
        for root, _dirs, files in os.walk(path):
            for fn in files:
                if fn.endswith(_PICKLE_SUFFIXES):
                    results.append(_builtin_scan_file(os.path.join(root, fn)))
        return _merge(results)
    return _builtin_scan_file(path)


# --- picklescan-backed path -----------------------------------------------------

def _picklescan_scan(path: str) -> PickleResult:
    try:
        from picklescan.scanner import scan_directory_path, scan_file_path
    except ImportError:  # pragma: no cover — fallback handles this
        return _builtin_scan(path)

    try:
        if os.path.isdir(path):
            res = scan_directory_path(path)
        else:
            res = scan_file_path(path)
    except Exception as exc:
        # picklescan raises on non-pickle files in some versions — treat as clean.
        return PickleResult(exploit="clean", tool="picklescan", reason=f"scan skipped: {exc}")

    findings: list[PickleFinding] = []
    for g in getattr(res, "globals", []) or []:
        safety = "innocuous"
        s = getattr(g, "safety", None)
        if s is not None:
            sval = getattr(s, "value", str(s))
            if sval == "dangerous":
                safety = "dangerous"
            elif sval == "suspicious":
                safety = "suspicious"
        findings.append(PickleFinding(module=getattr(g, "module", ""), name=getattr(g, "name", ""), safety=safety))

    scanned = int(getattr(res, "scanned_files", 0) or 0)
    infected = int(getattr(res, "infected_files", 0) or 0)
    dangerous = [f for f in findings if f.safety == "dangerous"]
    suspicious = [f for f in findings if f.safety == "suspicious"]

    if dangerous:
        names = ", ".join(sorted({f.qualname for f in dangerous}))
        return PickleResult(
            exploit="suspect", tool="picklescan", scanned_files=scanned, infected_files=infected,
            findings=findings, reason=f"dangerous code-exec globals: {names}",
        )
    if suspicious:
        names = ", ".join(sorted({f.qualname for f in suspicious}))
        return PickleResult(
            exploit="suspect", tool="picklescan", scanned_files=scanned, infected_files=infected,
            findings=findings, reason=f"suspicious globals: {names}",
        )
    # picklescan counts any touched file as "scanned" (including non-pickle
    # weight files), so distinguish via the globals list: a real pickle with
    # innocuous class refs yields non-empty globals -> pickle_found; a
    # safetensors/gguf yields no globals -> clean.
    if findings:
        return PickleResult(
            exploit="pickle_found", tool="picklescan", scanned_files=scanned, infected_files=infected,
            findings=findings, reason="pickle present, no dangerous globals",
        )
    return PickleResult(
        exploit="clean", tool="picklescan", scanned_files=0, findings=findings,
        reason="no __reduce__ / code-exec tensors found",
    )


def _merge(results: list[PickleResult]) -> PickleResult:
    if not results:
        return PickleResult(exploit="clean", tool="builtin", reason="no pickle files in directory")
    findings: list[PickleFinding] = []
    scanned = sum(r.scanned_files for r in results)
    infected = sum(r.infected_files for r in results)
    for r in results:
        findings.extend(r.findings)
    tool = results[0].tool
    if any(r.exploit == "suspect" for r in results):
        dangerous = [f for f in findings if f.safety in ("dangerous", "suspicious")]
        names = ", ".join(sorted({f.qualname for f in dangerous})) or "see findings"
        return PickleResult(exploit="suspect", tool=tool, scanned_files=scanned, infected_files=infected, findings=findings, reason=f"dangerous/suspicious globals: {names}")
    if findings:
        return PickleResult(exploit="pickle_found", tool=tool, scanned_files=scanned, infected_files=infected, findings=findings, reason="pickle(s) present, no dangerous globals")
    return PickleResult(exploit="clean", tool=tool, scanned_files=0, findings=findings, reason="no __reduce__ / code-exec tensors found")


def scan_path(path: str) -> PickleResult:
    """Run the exploit-detection pass on a file or directory.

    Uses ``picklescan`` if importable; otherwise the built-in ``pickletools``
    fallback. A non-pickle weight file (safetensors/gguf) yields ``clean``.
    """
    if not os.path.exists(path):
        return PickleResult(exploit="clean", tool="skipped", reason=f"path not found: {path}")
    try:
        import picklescan  # noqa: F401  (probe availability)
    except ImportError:
        return _builtin_scan(path)
    return _picklescan_scan(path)
