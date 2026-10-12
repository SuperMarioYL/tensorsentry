"""Release-pipeline contract tests.

The v0.2.0 release shipped with zero assets: the workflow built the wheel and
sdist but uploaded them only as a workflow artifact, never attaching them to
the GitHub release. These tests pin the attach contract so it cannot silently
regress. Parsing requires PyYAML (present on GitHub ubuntu runners; the test
skips where it is absent).
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _workflow_steps() -> list[dict]:
    wf = yaml.safe_load((_REPO_ROOT / ".github" / "workflows" / "release.yml").read_text())
    return wf["jobs"]["build"]["steps"]


def test_release_workflow_attaches_artifacts_to_the_release():
    steps = _workflow_steps()
    attach = [s for s in steps if "gh release upload" in s.get("run", "")]
    assert attach, (
        "release.yml must attach dist/* to the GitHub release — a tag push that "
        "only produces a workflow artifact ships a release with zero assets"
    )
    run = attach[0]["run"]
    assert "dist/" in run
    # Tolerate both orderings: release created by the publisher first, or not.
    assert "--clobber" in run
    assert "gh release create" in run
