"""Scan report model + rich-text / JSON rendering.

``ScanReport`` mirrors the plan's core data model — the three verdicts
(``structure`` / ``exploit`` / ``provenance``) plus the structured anomaly list.
It renders to a colored verdict block (rich) for the terminal and to a flat dict
for ``--json`` CI piping.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .pickle_scan import PickleResult
from .tensor_validate import StructureResult

__all__ = ["ScanReport", "render_rich", "to_json", "verdict_color"]


@dataclass
class ScanReport:
    """Full scan verdict for one artifact (file or directory)."""

    path: str
    model: str
    structure: str = "unknown_profile"  # "ok" | "anomaly" | "unknown_profile"
    exploit: str = "clean"  # "clean" | "suspect" | "pickle_found"
    provenance: str = "unreachable"  # "signed" | "unsigned" | "unreachable" (m3 stub)
    anomalies: list[str] = field(default_factory=list)
    structure_detail: dict[str, Any] = field(default_factory=dict)
    exploit_detail: dict[str, Any] = field(default_factory=dict)
    provenance_detail: dict[str, Any] = field(default_factory=dict)
    scanned_files: list[str] = field(default_factory=list)
    source_format: str = "safetensors"

    @property
    def ok(self) -> bool:
        """A scan is OK iff structure is ok (or unknown_profile) AND exploit is not suspect."""
        return self.structure in ("ok", "unknown_profile") and self.exploit != "suspect"

    @classmethod
    def from_results(
        cls,
        path: str,
        model: str,
        source_format: str,
        structure: StructureResult | None,
        exploit: PickleResult | None,
        scanned_files: list[str],
        provenance: str = "unreachable",
    ) -> "ScanReport":
        anomalies: list[str] = []
        structure_detail: dict[str, Any] = {}
        exploit_detail: dict[str, Any] = {}
        if structure is not None:
            anomalies.extend(str(a) for a in structure.anomalies)
            structure_detail = {
                "profile": structure.profile,
                "structure": structure.structure,
                "n_tensors": structure.n_tensors,
                "n_layers": structure.n_layers,
                "expert_count": structure.expert_count,
                "anomalies": [str(a) for a in structure.anomalies],
            }
        if exploit is not None:
            exploit_detail = {
                "exploit": exploit.exploit,
                "tool": exploit.tool,
                "scanned_files": exploit.scanned_files,
                "infected_files": exploit.infected_files,
                "findings": [
                    {"module": f.module, "name": f.name, "safety": f.safety}
                    for f in exploit.findings
                ],
                "reason": exploit.reason,
            }
        return cls(
            path=path,
            model=model,
            source_format=source_format,
            structure=structure.structure if structure else "unknown_profile",
            exploit=exploit.exploit if exploit else "clean",
            provenance=provenance,
            anomalies=anomalies,
            structure_detail=structure_detail,
            exploit_detail=exploit_detail,
            scanned_files=scanned_files,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def verdict_color(verdict: str, ok: bool = True) -> str:
    """Map a verdict word to a rich color."""
    if verdict in ("ok", "clean", "signed", "innocuous"):
        return "green"
    if verdict in ("anomaly", "suspect", "dangerous"):
        return "red"
    if verdict in ("pickle_found", "unsigned", "suspicious"):
        return "yellow"
    return "cyan"


def render_rich(report: ScanReport):
    """Return a ``rich.table.Table`` for the verdict block (terminal output)."""
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    tbl = Table(title=f"TensorSentry verdict — {report.model} [{report.source_format}]", show_header=True, header_style="bold cyan", expand=True)
    tbl.add_column("Check", style="bold")
    tbl.add_column("Status")
    tbl.add_column("Detail", overflow="fold")

    tbl.add_row(
        "STRUCTURE",
        Text(report.structure, style=verdict_color(report.structure, report.ok)),
        (
            f"{report.structure_detail.get('n_tensors', 0)} tensors, "
            f"{report.structure_detail.get('n_layers', 0)} layers, "
            f"experts={report.structure_detail.get('expert_count', 'n/a')}"
            if report.structure != "unknown_profile" else "no matching profile"
        ),
    )
    tbl.add_row(
        "EXPLOIT",
        Text(report.exploit, style=verdict_color(report.exploit, report.ok)),
        report.exploit_detail.get("reason", "n/a"),
    )
    tbl.add_row(
        "PROVENANCE",
        Text(report.provenance, style=verdict_color(report.provenance, report.ok)),
        "m3 milestone (stub) — not yet verified" if report.provenance == "unreachable" else report.provenance,
    )

    verdict_word = "OK" if report.ok else "FLAG"
    verdict_style = "bold green" if report.ok else "bold red"
    banner = Text(f"\n  {report.path}\n  → overall: {verdict_word}\n", style=verdict_style)

    if report.anomalies:
        body = []
        for a in report.anomalies:
            body.append(Text(f"  ! {a}", style="yellow"))
        return Group(tbl, banner, Panel(Group(*body), title="anomalies", border_style="yellow"))
    return Group(tbl, banner)


def to_json(report: ScanReport) -> str:
    """Render the report as a flat JSON string for ``--json`` CI piping."""
    return json.dumps(report.to_dict(), indent=2, sort_keys=False)
