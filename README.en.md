<div align="right"><sub><b>English</b>&nbsp;&nbsp;⇄&nbsp;&nbsp;<a href="./README.md">中文</a></sub></div>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/hero-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="./assets/hero-light.svg">
  <img src="./assets/hero-light.svg" width="880" alt="TensorSentry — agent-safety scanner for CN-model weights">
</picture>

<p align="center"><sub>Validate MoE/MLA tensor structure + detect pickle exploits in DeepSeek-V4 / Kimi K3 / Qwen3.7 weight artifacts, before agents load them.</sub></p>

**Block poisoned CN-model weights before your agent calls `load_model()`.** The 2025 Hugging Face intrusion proved the registry gate is not enough, and picklescan cannot tell a DeepSeek MLA weight from a generic pickle — TensorSentry validates each CN model's MoE/MLA tensor structure and detects pickle/code-exec exploits before load.

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="license"></a>
  <a href="https://github.com/SuperMarioYL/tensorsentry/releases"><img src="https://img.shields.io/github/v/release/SuperMarioYL/tensorsentry?color=%235E5CE6&label=release" alt="release"></a>
  <a href="https://github.com/SuperMarioYL/tensorsentry/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/SuperMarioYL/tensorsentry/ci.yml?branch=main&label=ci" alt="ci"></a>
  <img src="https://img.shields.io/badge/python-%E2%89%A53.10-3776AB.svg" alt="python">
  <img src="https://img.shields.io/badge/Agent--safety-weights%20as%20attack%20surface-5E5CE6.svg" alt="Agent-safety">
</p>

## Contents
- [Architecture](#architecture)
- [Why this exists](#why-this-exists)
- [Install & Quickstart](#install--quickstart)
- [Usage](#usage)
- [Demo](#demo)
- [vs picklescan](#vs-picklescan)
- [Roadmap](#roadmap)
- [License](#license)
- [Share](#share)

<h2><img src="https://api.iconify.design/tabler:topology-star-3.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Architecture</h2>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/atlas-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="./assets/atlas-light.svg">
  <img src="./assets/atlas-light.svg" width="880" alt="architecture: weight artifact → TensorSentry scan (header parse / profile match / pickle scan) → verdict">
</picture>

One Python process, one CLI. The readers parse only the header (tensor names/shapes/dtypes) and never load full weights, so a 70 GB checkpoint scans in seconds. The core primitive is the `TensorProfile` — a declarative, per-model MoE/MLA tensor-structure schema that generic pickle scanners structurally lack and have no reason to encode.

```
cli.py ──► scanner.py (orchestrator)
             ├─► safetensors_reader.py / gguf_reader.py   (header-only parse, no full load)
             ├─► tensor_validate.py ◄── model_profiles/{deepseek_v4,kimi_k3,qwen3_7}.py
             ├─► pickle_scan.py    (wraps picklescan lib — do not reinvent)
             └─► provenance.py     (ModelScope/HF API + sigstore verify — m3 stub)
          ──► report.py    (rich text + JSON)
```

<h2><img src="https://api.iconify.design/tabler:shield-check.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Why this exists</h2>

DeepSeek-V4 / Kimi K3 / Qwen3.7 ship weekly via ModelScope, and agent runtimes default to `load_model()`-ing them into a tool-calling loop. The 2025 Hugging Face intrusion ([the 500-upvote HN post where Tailscale concedes it didn't stop it](https://tailscale.com/blog/hugging-face-intrusion)) made "weights are an attack surface" a lived event — but picklescan reads arbitrary pickle bytes for code-exec and cannot assert "this DeepSeek MLA weight must contain `q_lora_rank=1536`/`kv_lora_rank=512` projection tensors", let alone that a Kimi K3 must expose 896 experts. **The per-model structural layer before agent load is exactly the layer CN mirrors are missing post-intrusion.**

<h2><img src="https://api.iconify.design/tabler:rocket.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Install & Quickstart</h2>

```bash
pip install tensorsentry                                  # or uv tool install tensorsentry
tensorsentry profiles                                      # list supported CN-model profiles
tensorsentry scan --model deepseek-v4 ./model.safetensors # validate structure + detect exploits before load
```

<details><summary>Run from source (development)</summary>

```bash
git clone https://github.com/SuperMarioYL/tensorsentry && cd tensorsentry
pip install -e .
pytest -q
```
</details>

<h2><img src="https://api.iconify.design/tabler:terminal-2.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Usage</h2>

Three subcommands cover m1 (structure validation) and m2 (combined structure + exploit scan):

```bash
# list registered CN-model profiles (MLA / MoE structural schemas)
tensorsentry profiles

# m1 — structure-only: DeepSeek-V4 MLA projections + 256-expert MoE
tensorsentry validate deepseek-v4 ./model-00001-of-0000X.safetensors

# m2 — combined structure + exploit scan (.gguf and full model dirs supported too)
tensorsentry scan --model deepseek-v4 ./deepseek-v4.gguf
tensorsentry scan --model kimi-k3 ./kimi-k3/          # scan a whole ModelScope pull dir
tensorsentry scan ./suspect_dir                       # no model → exploit-only

# CI gating: --json output + non-zero exit on a flag
tensorsentry scan --model qwen3.7 ./qwen.safetensors --json
```

<details><summary>Sample combined verdict output</summary>

```
                      TensorSentry verdict — deepseek-v4 [safetensors]
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Check         ┃ Status        ┃ Detail                                       ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ STRUCTURE     │ ok            │ 778 tensors, 2 layers, experts=256           │
│ EXPLOIT       │ clean         │ no __reduce__ / code-exec tensors found      │
│ PROVENANCE    │ unreachable   │ m3 milestone (stub) — not yet verified       │
└───────────────┴───────────────┴──────────────────────────────────────────────┘
```
</details>

<h2><img src="https://api.iconify.design/tabler:photo.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Demo</h2>

![demo](assets/demo.gif)

The 10-minute happy path from `profiles` → `validate` → `scan` → `--json` (`docs/demo.tape`, rendered to gif by CI via vhs).

<h2><img src="https://api.iconify.design/tabler:arrows-exchange-2.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> vs picklescan</h2>

| Axis | [picklescan](https://github.com/mmaitre314/picklescan) | TensorSentry |
|---|---|---|
| pickle code-exec detection | ✓ (arbitrary pickle bytes) | ✓ (wraps picklescan, no reinvention) |
| per-model MoE/MLA tensor-structure validation | — | ✓ DeepSeek MLA ranks + Kimi 896 experts |
| .safetensors / .gguf header-only scan | partial | ✓ 70 GB checkpoint in seconds, no weights loaded |
| ModelScope/HF distribution awareness | — | ✓ the layer CN mirrors are missing |
| maintenance cost | generic, substrate-agnostic | must track each CN model's weekly tensor schema |

picklescan does one thing well — scanning arbitrary pickle bytes for code-exec. TensorSentry's irreducible novelty is the **per-model structural validation** generic scanners structurally cannot do (the table's last row: tracking per-model schemas opposes their substrate-agnostic philosophy).

<h2><img src="https://api.iconify.design/tabler:map-2.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Roadmap</h2>

- [x] **m1** — safetensors reader + tensor_validate + deepseek-v4 MLA profile; `validate` reports structural anomalies
- [x] **m2** — pickle_scan (wraps picklescan) + gguf_reader + kimi-k3 / qwen3.7 profiles; `scan` emits combined structure+exploit verdict
- [ ] **m3** — provenance (ModelScope/HF manifest + sigstore) + report polish + PyPI publish + Gitee mirror; end-to-end `pip install tensorsentry` ready for Show HN
- [ ] v0.2 — agent-runtime `load_model()` preflight hook + CI gate plugin (currently `out_of_scope`, the next wedge)

<h2><img src="https://api.iconify.design/tabler:license.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> License</h2>

MIT, see [LICENSE](./LICENSE). Issues / PRs welcome at [github.com/SuperMarioYL/tensorsentry/issues](https://github.com/SuperMarioYL/tensorsentry/issues). After pushing, set repo topics: `gh repo edit --add-topic agent-safety --add-topic supply-chain --add-topic safetensors`.

## Share

```
TensorSentry — the agent-safety scanner that checks CN-model MoE/MLA tensor structure + pickle exploit before your agent loads the weight. HF intrusion proved weights are the attack surface; picklescan can't tell a DeepSeek MLA weight from a pickle. https://github.com/SuperMarioYL/tensorsentry
```

<p align="center"><sub><a href="./LICENSE">MIT</a> © 2026 SuperMarioYL</sub></p>
