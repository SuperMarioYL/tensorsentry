"""TensorSentry CLI — ``tensorsentry`` (click + rich).

Commands:
    profiles                list supported CN-model profiles
    validate MODEL PATH    m1: validate a weight artifact's MoE/MLA structure
    scan PATH               m2: combined structure + exploit (+ optional provenance)

Run ``tensorsentry scan --help`` for the option surface.
"""

from __future__ import annotations

import json as _json
import sys
from typing import Iterable

import click
from rich.console import Console

from . import __version__
from .model_profiles import list_profiles
from .report import ScanReport, render_rich, to_json
from .scanner import known_profiles, scan as run_scan, validate as run_validate

console = Console()


def _emit(report: ScanReport, as_json: bool, quiet: bool) -> None:
    if as_json:
        click.echo(to_json(report))
        return
    if quiet:
        click.echo(f"{report.path}\t{report.model}\t{report.structure}\t{report.exploit}\t{report.provenance}")
        return
    console.print(render_rich(report))


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="tensorsentry")
def cli() -> None:
    """TensorSentry — agent-safety scanner for CN-model weight artifacts.

    Validates per-model MoE/MLA tensor structure and detects pickle/code-exec
    exploits inside DeepSeek-V4 / Kimi K3 / Qwen3.7 .safetensors & .gguf
    checkpoints, before an agent runtime loads them.
    """


@cli.command("profiles")
def profiles_cmd() -> None:
    """List the supported CN-model profiles."""
    tbl_rows = []
    for p in list_profiles():
        mla = (
            f"q_lora_rank={p.mla.q_lora_rank} kv_lora_rank={p.mla.kv_lora_rank} n_heads={p.mla.n_heads}"
            if p.mla else "—"
        )
        moe = f"n_experts={p.moe.n_experts} shared={p.moe.n_shared}" if p.moe else "—"
        tbl_rows.append((p.model_id, p.architecture, mla, moe))
    if not tbl_rows:
        click.echo("(no profiles registered)")
        return
    # Plain, terminal-friendly table (works without rich rendering quirks).
    click.echo(f"{'model':<14}{'architecture':<42}{'MLA':<46}{'MoE'}")
    click.echo("-" * 120)
    for mid, arch, mla, moe in tbl_rows:
        click.echo(f"{mid:<14}{arch[:40]:<42}{mla[:44]:<46}{moe}")


@cli.command("validate")
@click.argument("model")
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@click.option("--quiet", "-q", is_flag=True, help="One-line TSV verdict (CI grepping).")
def validate_cmd(model: str, path: str, as_json: bool, quiet: bool) -> None:
    """m1: validate a weight artifact's tensor structure against MODEL.

    \b
    Example:
      tensorsentry validate deepseek-v4 ./model-00001-of-0000X.safetensors
    """
    report = run_validate(model, path)
    _emit(report, as_json, quiet)
    if not report.ok and report.structure != "unknown_profile":
        sys.exit(1)


@cli.command("scan")
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.option("--model", "-m", default=None, help="Profile id, e.g. deepseek-v4 / kimi-k3 / qwen3.7. Omit to scan exploits only.")
@click.option("--provenance", is_flag=True, help="Also attempt provenance verification (m3 — currently a stub).")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@click.option("--quiet", "-q", is_flag=True, help="One-line TSV verdict (CI grepping).")
def scan_cmd(path: str, model: str | None, provenance: bool, as_json: bool, quiet: bool) -> None:
    """m2: combined structure + exploit scan of a weight artifact.

    \b
    Examples:
      tensorsentry scan --model deepseek-v4 ./model-00001-of-0000X.safetensors
      tensorsentry scan --model kimi-k3 ./kimi-k3.gguf --json
      tensorsentry scan ./suspect_dir            # exploit-only if no model given
    """
    report = run_scan(path, model_id=model, include_provenance=provenance)
    _emit(report, as_json, quiet)
    # Exit non-zero on a real flag (suspect exploit OR structural anomaly).
    if report.exploit == "suspect" or report.structure == "anomaly":
        sys.exit(1)


def main(argv: Iterable[str] | None = None) -> int:
    """Console-script entrypoint."""
    try:
        cli.main(args=list(argv) if argv is not None else None, standalone_mode=False)
    except click.exceptions.UsageError as exc:
        click.echo(str(exc), err=True)
        exc.show()
        return 2
    except click.exceptions.Abort:
        return 130
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
