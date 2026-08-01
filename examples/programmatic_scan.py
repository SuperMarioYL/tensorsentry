"""Minimal programmatic API example — scan a weight artifact from Python.

    from tensorsentry.scanner import scan
    report = scan("./model.safetensors", model_id="deepseek-v4")
    print(report.structure, report.exploit, report.anomalies)
"""

from tensorsentry.scanner import scan
from tensorsentry.report import to_json

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "./model.safetensors"
    model = sys.argv[2] if len(sys.argv) > 2 else "deepseek-v4"
    report = scan(path, model_id=model)
    print(to_json(report))
    sys.exit(0 if report.ok else 1)
