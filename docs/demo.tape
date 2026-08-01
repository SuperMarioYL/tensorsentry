Output "assets/demo.gif"
Set FontFamily "JetBrains Mono"
Set FontSize 13
Set Width 1100
Set Height 600
Set Padding 16
Set Theme "Dracula"

# Prepare a tiny DeepSeek-V4-shaped safetensors fixture (header-only, ~150 KB).
Type "python examples/make_fixture.py /tmp/ts-demo.safetensors"
Enter
Sleep 600ms

# Step 1 — list the supported CN-model profiles.
Type "tensorsentry profiles"
Enter
Sleep 700ms

# Step 2 — m1: validate the artifact's MLA + MoE tensor structure.
Type "tensorsentry validate deepseek-v4 /tmp/ts-demo.safetensors"
Enter
Sleep 1200ms

# Step 3 — m2: combined structure + exploit scan.
Type "tensorsentry scan --model deepseek-v4 /tmp/ts-demo.safetensors"
Enter
Sleep 1400ms

# JSON output for CI piping.
Type "tensorsentry scan --model deepseek-v4 /tmp/ts-demo.safetensors --json | head -20"
Enter
Sleep 1200ms
