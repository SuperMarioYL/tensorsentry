# Changelog

All notable changes to TensorSentry are documented here. Versions follow
[semver](https://semver.org/); the format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] — 2026-09-09

### Fixed

- **Corrupt or malformed weight artifacts now fail the scan instead of passing
  silently or crashing.** Previously a `.safetensors`/`.gguf` file that claimed
  a known weight format but could not be parsed (truncated download, tampered
  header) produced `structure: unknown_profile` with no anomalies and exit
  code 0 from both `tensorsentry scan` and `tensorsentry validate`; a GGUF
  with a string-typed `general.alignment` field crashed with a raw
  `ValueError` traceback. Reader errors are now surfaced as a `reader_error`
  structure anomaly and the CLI exits non-zero, while paths that are not
  weight containers at all keep the exploit-only fallback (a mislabelled
  pickle is still a real attack shape).
- **Deterministic directory handling + GGUF directory support.** A directory
  containing both `.safetensors` and `.gguf` files picked its format by
  arbitrary `os.listdir` order, and directories of `.gguf` files could not be
  structurally scanned at all. Format detection is now deterministic
  (safetensors first) and all `*.gguf` files in a directory are merged and
  validated in one pass.
- **Built-in pickle fallback now detects modern pickles.** When the
  `picklescan` library is unavailable, the built-in `pickletools` fallback
  failed to resolve `STACK_GLOBAL` operands (used by every Python 3 default
  protocol-2+ pickle), so `os.system` code-exec gadgets were classified as
  innocuous and malicious pickles verdicted `pickle_found` instead of
  `suspect`. Operands are now resolved by walking back through string pushes
  and the memo (the same technique the library uses); stack-computed operands
  are treated as suspicious rather than trusted.
- **Fallback robustness:** distinct globals from the same module are no longer
  deduplicated into one finding; non-pickle files are no longer read fully
  into memory before being verdicted clean; the fallback directory scan now
  also covers `.dat`/`.data`/`.joblib`/`.ckpt` files like the primary path.
- **MLA layer completeness.** A transformer layer whose entire MLA attention
  block was stripped (MoE left intact) validated as `ok`, contradicting the
  documented per-layer completeness contract. Such layers are now flagged with
  a `missing_mla` anomaly; sharding tolerance is unchanged (only whole missing
  layers are tolerated).

### Changed

- Pinned `picklescan>=0.0.21,<2` — the wrapper is verified against the 1.x
  API surface; future majors are opted into deliberately rather than by
  dependency drift.

## [0.1.0] — 2026-08-01

### Added

- Initial release: per-model MoE/MLA tensor-structure validation for
  DeepSeek-V4, Kimi K3 and Qwen3.7 against `.safetensors` and `.gguf`
  checkpoints (header-only parsing), pickle/code-exec exploit detection
  wrapping `picklescan`, and a `scan`/`validate`/`profiles` CLI with rich and
  JSON output.
