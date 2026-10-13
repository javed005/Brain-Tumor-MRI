"""Package verified training outputs and code for upload after a real run."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def package(artifacts, output):
    project = Path(__file__).resolve().parent
    artifacts = Path(artifacts).resolve()
    completed = artifacts / "COMPLETED.json"
    if not completed.is_file():
        raise ValueError("Training is not complete: COMPLETED.json is missing.")
    status = json.loads(completed.read_text())
    if status.get("software_test_mode") or status.get("pretrained_weights") != "imagenet":
        raise ValueError("Random-weight software tests cannot be packaged as an assignment result. Run with --weights imagenet.")
    expected = {"custom_cnn", "mobilenet_v2", "efficientnet_b0"}
    if set(status["models"]) != expected:
        raise ValueError("The full submission needs the custom CNN and both transfer models. Train all three.")
    required = [artifacts / "deployment" / "model.keras", artifacts / "deployment" / "metadata.json",
                artifacts / "results" / "REPORT.md", artifacts / "results" / "model_comparison.csv",
                artifacts / "class_names.json"]
    required += [artifacts / "models" / f"{name}.h5" for name in sorted(expected)]
    required += [artifacts / "results" / f"{name}_test_metrics.json" for name in sorted(expected)]
    if any(not p.is_file() for p in required):
        raise ValueError("Some required training/model/evaluation files are missing.")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(project.iterdir()):
            if path.is_file() and path.suffix in {".py", ".md", ".txt", ".ipynb", ".json"}:
                archive.write(path, Path("Brain_Tumor_MRI_Submission") / path.name)
        ignore = {".keras"}  # Avoid storing each model twice; selected deployment stays.
        for path in sorted(artifacts.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(artifacts)
            if relative.parts[0] == "models" and path.suffix in ignore:
                continue
            archive.write(path, Path("Brain_Tumor_MRI_Submission") / "artifacts" / relative)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--output", default="Brain_Tumor_MRI_Trained_Submission.zip")
    args = parser.parse_args()
    try:
        print("Submission ZIP created:", package(args.artifacts, args.output))
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}")
