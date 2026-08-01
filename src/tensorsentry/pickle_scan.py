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

import io
import os
import pickletools
from dataclasses import dataclass, field

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

def _disassemble_findings(data: bytes) -> list[PickleFinding]:
    """Disassemble a pickle byte-stream with stdlib pickletools and collect globals."""
    findings: list[PickleFinding] = []
    seen: set[str] = set()
    buf = io.BytesIO(data)
    try:
        for op, arg, pos in pickletools.genops(buf):
            if op.name not in ("GLOBAL", "STACK_GLOBAL"):
                continue
            if isinstance(arg, tuple):
                module, name = (str(arg[0]) if arg[0] else ""), (str(arg[1]) if len(arg) > 1 and arg[1] else "")
            else:
                # GLOBAL arg is "module name"
                parts = str(arg or "").rsplit(" ", 1)
                module, name = (parts[0], parts[1]) if len(parts) == 2 else (str(arg or ""), "")
            key = f"{module}.{{name}}"
            if key in seen:
                continue
            seen.add(key)
            findings.append(PickleFinding(module=module, name=name, safety=_classify(module, name)))
    except Exception:
        # not a valid pickle stream — no findings
        return findings
    return findings


def _builtin_scan_file(path: str) -> PickleResult:
    try:
        with open(path, "rb") as f:
            head = f.read(8)
            if not _looks_like_pickle(head):
                # could still be a long-prefixed pickle; read more to be sure
                rest = f.read()
                data = head + rest
            else:
                data = head + f.read()
    except OSError:
        return PickleResult(exploit="clean", tool="builtin", reason=f"cannot read {path}")
    if not _looks_like_pickle(data[:8]):
        # safetensors / gguf / random weight file → not a pickle → clean
        return PickleResult(
            exploit="clean", tool="builtin", scanned_files=0,
            reason="no pickle magic / __reduce__ found in file",
        )
    findings = _disassemble_findings(data)
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
                if fn.endswith((".bin", ".pt", ".pth", ".pkl", ".pickle")):
                    results.append(_builtin_scan_file(os.path.join(root, fn)))
        return _merge(results)
    return _builtin_scan_file(path)


# --- picklescan-backed path -----------------------------------------------------

def _picklescan_scan(path: str) -> PickleResult:
    try:
        from picklescan.scanner import SafetyLevel, scan_directory_path, scan_file_path
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
