[English](./README.en.md) · [Website](https://tensorsentry.lei6393.com) · [GitHub](https://github.com/SuperMarioYL/tensorsentry)

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/hero-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/hero-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/hero-dark.svg">
  <img src="./assets/presentation/hero-light.svg" width="960" alt="Hero diagram">
</picture>

# tensorsentry

**加载检查点之前先检查张量结构。**

TensorSentry 读取张量名称、dtype 与 shape，与声明配置比较并报告结构异常。

## 为什么需要它

检查点缺少投影或 shape 不匹配，可能直到加载时才报错。元数据检查可以在主加载链路分配模型张量前暴露这些差异。

- **先检查元数据** — shape 检查无需执行推理。
- **声明期望结构** — 配置明确必须满足的张量约束。
- **区分检查结果** — 结构、攻击与来源状态各自表达不同含义。

## 架构

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/architecture-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/architecture-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/architecture-dark.svg">
  <img src="./assets/presentation/architecture-light.svg" width="960" alt="Architecture diagram">
</picture>

容器读取器提供统一张量视图。结构验证器应用 TensorProfile 要求，包括声明的 MoE 与 MLA 约束。scanner 将结构结果与 pickle 扫描适配器组合；provenance 当前是独立的未实现阶段。

| 组件 | 职责 |
| --- | --- |
| `Header reader` | safetensors_reader / gguf_reader |
| `TensorProfile` | Declared names and shapes |
| `Structure validator` | tensor_validate.py |
| `Combined report` | scanner.py; report.py |

## 安装与快速上手

使用仓库清单指定的运行时版本构建，并在仓库根目录运行示例。

```bash
git clone https://github.com/SuperMarioYL/tensorsentry.git
cd tensorsentry
uv venv .venv
uv pip install --python .venv/bin/python -e .
source .venv/bin/activate
```

在临时目录生成微型 safetensors 文件，检查明确的 shape 配置，再展示空输入的缺失张量拒绝。

```bash
PYTHONPATH=src python3 examples/presentation-demo.py
```

## 实际运行示例

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/process-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/process-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/process-dark.svg">
  <img src="./assets/presentation/process-light.svg" width="960" alt="Process diagram">
</picture>

The real tiny tensor passes its demo profile; an empty tensor set produces a missing-tensor anomaly.

```text
{
  "fixture_profile": "demo-linear",
  "tensor_count": 1,
  "valid_structure": "ok",
  "empty_structure": "anomaly",
  "missing_codes": [
    "missing_tensor"
  ]
}
```

完整命令与输出保存在 [docs/demo-results.json](./docs/demo-results.json). 输入和复现代码均随仓提供。

![已有终端录制](./assets/demo.gif)

保留已有录制供参考；上方文字示例给出当前可复现的操作。

## 用法

CLI 提供以下操作。示例之外的命令需要替换成你的文件路径或标识。

```bash
tensorsentry profiles
tensorsentry validate <profile-id> model.safetensors
tensorsentry scan --model <profile-id> ./checkpoint --json
```

## 配置

注册配置应在与实际模型假设核对后选择。Python API 接受显式 TensorProfile，示例用它检查含两个 F32 值的张量。头解析器不会把模型加载进推理运行时。

## 集成与职责分工

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/integrations-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/integrations-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/integrations-dark.svg">
  <img src="./assets/presentation/integrations-light.svg" width="960" alt="Integrations diagram">
</picture>

以下路径已有源码实现。按任务选择输入，并把生成的结果与项目一起保存。

| 路径 | 已实现职责 |
| --- | --- |
| safetensors | Header metadata parsing |
| GGUF | Tensor metadata adapter |
| TensorProfile | Declared structural rules |
| picklescan | Exploit-scanning integration |
| JSON report | Structured scan findings |

## 限制与后续方向

- 配置是仓库中的声明，需要与实际检查点架构核对；配置名称不等于当前厂商规格已获验证。
- 结构通过不证明来源、数值正确性或不存在所有攻击；来源验证尚未实现。
- 示例使用演示专用配置与微型有效张量文件，不评估真实模型权重或 pickle 检测。

来源验证与运行时集成属于后续工作，新增配置需要有代表性的检查点证据。

## 许可与贡献

许可见 [LICENSE](./LICENSE). 反馈问题时请提供最小输入、执行命令和实际输出。
