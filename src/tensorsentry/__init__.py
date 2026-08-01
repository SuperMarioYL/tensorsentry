"""TensorSentry — agent-safety scanner for CN-model weight artifacts.

Validates per-model MoE/MLA tensor structure and detects pickle/code-exec
exploits inside DeepSeek-V4 / Kimi K3 / Qwen3.7 ``.safetensors`` and ``.gguf``
checkpoints *before* an agent runtime loads them.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
