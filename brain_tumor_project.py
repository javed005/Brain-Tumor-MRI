"""Train, evaluate, compare and export brain MRI classification models.

Run: python brain_tumor_project.py --data dataset --output artifacts
Results are calculated from the supplied images; no performance is hardcoded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import platform
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from mri_utils import CLASS_NAMES, DATASET_URL, IMAGE_EXTENSIONS
from mri_utils import discover_dataset_root, load_image_array, write_json

MODEL_NAMES = ("custom_cnn", "mobilenet_v2", "efficientnet_b0")
tf = None  # Imported only when training starts, so data inspection needs no GPU.


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def scan_dataset(root):
    """Validate images and preserve the provided split assignments."""
    children = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    split_dirs = {
        "train": children["train"],
        "validation": next(children[k] for k in ("valid", "validation", "val") if k in children),
        "test": children["test"],
    }
    records, rejected = [], []
    for split, split_dir in split_dirs.items():
        folders = {p.name.lower(): p for p in split_dir.iterdir() if p.is_dir()}
        missing = set(CLASS_NAMES) - set(folders)
        if missing:
            raise ValueError(f"{split}: missing class folders {sorted(missing)}")
        for label, class_name in enumerate(CLASS_NAMES):
            for path in sorted(folders[class_name].rglob("*")):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                relative = path.relative_to(root).as_posix()
                try:
                    with Image.open(path) as image:
                        image = ImageOps.exif_transpose(image).convert("RGB")
                        image.load()
                        width, height = image.size
                        digest = hashlib.sha256(
                            f"{width},{height},RGB:".encode() + image.tobytes()
                        ).hexdigest()
                    # Roboflow names identify transformed versions of one source.
                    family = path.name.split(".rf.")[0] if ".rf." in path.name else None
                    records.append(dict(
                        path=str(path), relative_path=relative, split=split,
                        class_name=class_name, label=label, width=width, height=height,
                        pixel_hash=digest, source_family=family,
                    ))
                except (OSError, ValueError, Image.DecompressionBombError) as exc:
                    rejected.append(dict(path=relative, reason=f"unreadable: {exc}"))
    if not records:
        raise ValueError("No readable class-labelled images were found.")
    return records, rejected


def remove_leakage(records):
    """Union exact pixel duplicates and named source families, then keep one.

    Priority is test, validation, train. A conflicting-label family is removed
    entirely. Keeping one member per family also avoids repeated-source weighting.
    This does not establish patient independence or detect all near duplicates.
    """
    parents = list(range(len(records)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(i, j):
        parents[find(i)] = find(j)

    seen_hash, seen_family = {}, {}
    for i, record in enumerate(records):
        for value, seen in ((record["pixel_hash"], seen_hash), (record["source_family"], seen_family)):
            if value is None:
                continue
            if value in seen:
                union(i, seen[value])
            else:
                seen[value] = i
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[find(i)].append(record)
    kept, rejected = [], []
    priority = {"test": 0, "validation": 1, "train": 2}
    for group in groups.values():
        if len({r["label"] for r in group}) > 1:
            rejected.extend(dict(path=r["relative_path"], reason="conflicting labels in source/duplicate family") for r in group)
            continue
        group.sort(key=lambda r: (priority[r["split"]], r["relative_path"]))
        kept.append(group[0])
        rejected.extend(dict(
            path=r["relative_path"],
            reason=f"duplicate/source family; retained {group[0]['relative_path']}"
        ) for r in group[1:])
    kept.sort(key=lambda r: (r["split"], r["label"], r["relative_path"]))
    for split in ("train", "validation", "test"):
        for name in CLASS_NAMES:
            if not any(r["split"] == split and r["class_name"] == name for r in kept):
                raise ValueError(f"Cleaning left no {name} images in {split}. Check the source data.")
    return kept, rejected


def make_dataset(records, size, batch_size, training, seed):
    paths = [r["path"] for r in records]
    labels = [r["label"] for r in records]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(records), seed=seed, reshuffle_each_iteration=True)

    def decode(path, label):
        def read(path_bytes):
            if isinstance(path_bytes, np.ndarray):
                path_bytes = path_bytes.item()
            return load_image_array(path_bytes.decode("utf-8"), size)
        image = tf.numpy_function(read, [path], tf.float32)
        image.set_shape((size, size, 3))
        return image, tf.cast(label, tf.int32)

    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    if training:
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.04, fill_mode="reflect", seed=seed + 1),
            tf.keras.layers.RandomZoom(0.08, fill_mode="reflect", seed=seed + 2),
            tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="reflect", seed=seed + 3),
            tf.keras.layers.RandomBrightness(0.08, value_range=(0, 255), seed=seed + 4),
        ], name="training_augmentation")
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    # No decoded-image cache: prevents large 224x224 RGB datasets filling RAM.
    return ds.prefetch(tf.data.AUTOTUNE)


def build_model(name, size, weights="imagenet"):
    layers = tf.keras.layers
    inputs = layers.Input(shape=(size, size, 3), name="raw_rgb_pixels")
    base = None
    if name == "custom_cnn":
        x = layers.Rescaling(1 / 255.0, name="normalization_0_to_1")(inputs)
        for index, filters in enumerate((32, 64, 128, 256)):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.MaxPooling2D()(x)
            if index in (1, 3):
                x = layers.Dropout(0.15)(x)
        x = layers.GlobalAveragePooling2D()(x)
    elif name == "mobilenet_v2":
        # MobileNetV2 requires [-1,1], embedded in the saved model.
        x = layers.Rescaling(1 / 127.5, offset=-1.0, name="mobilenet_normalization")(inputs)
        base = tf.keras.applications.MobileNetV2(
            include_top=False, weights=weights, input_shape=(size, size, 3)
        )
        base.trainable = False
        x = layers.GlobalAveragePooling2D()(base(x, training=False))
    elif name == "efficientnet_b0":
        # EfficientNet already includes its own normalization: input stays 0..255.
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=weights, input_shape=(size, size, 3)
        )
        base.trainable = False
        x = layers.GlobalAveragePooling2D()(base(inputs, training=False))
    else:
        raise ValueError(f"Unknown model: {name}")
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax", name="class_probabilities")(x)
    return tf.keras.Model(inputs, outputs, name=name), base


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"],
    )


def callbacks(checkpoint):
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def calculate_metrics(labels, probabilities):
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=list(range(4)), average="macro", zero_division=0
    )
    return dict(
        accuracy=float(accuracy_score(labels, predictions)),
        macro_precision=float(precision), macro_recall=float(recall), macro_f1=float(f1),
        classification_report=classification_report(
            labels, predictions, labels=list(range(4)), target_names=CLASS_NAMES,
            output_dict=True, zero_division=0,
        ),
        confusion_matrix=confusion_matrix(labels, predictions, labels=list(range(4))).tolist(),
    )


def plots_for_data(records, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    counts = pd.DataFrame([
        dict(split=s, class_name=c, images=sum(r["split"] == s and r["class_name"] == c for r in records))
        for s in ("train", "validation", "test") for c in CLASS_NAMES
    ])
    counts.to_csv(output / "class_counts.csv", index=False)
    ax = counts.pivot(index="class_name", columns="split", values="images").plot.bar(figsize=(9, 5))
    ax.set_ylabel("Images after duplicate/source cleaning")
    ax.set_xlabel("Class")
    ax.set_title("Dataset class distribution")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output / "class_distribution.png", dpi=170)
    plt.close()
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for column, name in enumerate(CLASS_NAMES):
        samples = [r for r in records if r["class_name"] == name and r["split"] == "train"][:2]
        for row in range(2):
            axes[row, column].axis("off")
            if row < len(samples):
                with Image.open(samples[row]["path"]) as image:
                    axes[row, column].imshow(ImageOps.exif_transpose(image).convert("RGB"))
                axes[row, column].set_title(name.replace("_", " "))
    fig.tight_layout()
    fig.savefig(output / "sample_images.png", dpi=150)
    plt.close(fig)


def plot_model(history_rows, matrix, name, output):
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import ConfusionMatrixDisplay
    frame = pd.DataFrame(history_rows)
    frame.to_csv(output / f"{name}_history.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ("accuracy", "loss")):
        axis.plot(frame["epoch"], frame[metric], label="Training")
        axis.plot(frame["epoch"], frame[f"val_{metric}"], label="Validation")
        axis.set(xlabel="Epoch", ylabel=metric.capitalize(), title=f"{name}: {metric}")
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / f"{name}_training_curves.png", dpi=170)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(np.array(matrix), display_labels=CLASS_NAMES).plot(
        ax=axis, cmap="Blues", colorbar=False, xticks_rotation=25,
    )
    axis.set_title(f"{name}: held-out test confusion matrix")
    fig.tight_layout()
    fig.savefig(output / f"{name}_confusion_matrix.png", dpi=170)
    plt.close(fig)


def generate_report(output, comparisons, selected, summary):
    """Write an honest report, using only metrics produced by the current run."""
    import pandas as pd
    text = [
        "# Brain Tumor MRI Image Classification", "",
        "## Objective", "",
        "Compare a custom convolutional network with ImageNet transfer learning for "
        "four MRI image classes and provide a Streamlit research demonstration.", "",
        "## Dataset and cleaning", "",
        f"Assigned source: {DATASET_URL}", "",
        "Classes: glioma, meningioma, no_tumor, pituitary.", "",
        f"Readable source images: {summary['readable_before_cleaning']}. "
        f"Retained: {summary['retained_images']}. Excluded: {summary['excluded_images']}.", "",
        "Exact decoded-pixel duplicates and Roboflow source-name families were grouped. "
        "One image per family was retained, prioritizing test, validation, then train. "
        "Conflicting-label families were excluded. See data_manifest.csv and excluded_images.csv.", "",
        "## Method", "",
        "Images are decoded as RGB, corrected for EXIF orientation, and resized with PIL bilinear "
        "interpolation. Training uses mild rotation, zoom, translation, horizontal flipping and "
        "brightness changes. Validation and test images are never augmented. "
        "Normalization is embedded in each saved model: custom CNN 0–1, MobileNetV2 −1–1, "
        "and EfficientNetB0's built-in preprocessing on raw 0–255 input.", "",
        "Balanced class weights, dropout, batch normalization, early stopping, learning-rate "
        "reduction and best-validation-loss checkpoints are used. The transfer backbones are "
        "initially frozen; optional final-layer fine-tuning keeps batch normalization frozen.", "",
        "## Model comparison", "",
        "| Model | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 | Parameters | Inference ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in comparisons:
        text.append(f"| {r['model']} | {r['validation_accuracy']:.4f} | {r['validation_macro_f1']:.4f} | "
                    f"{r['test_accuracy']:.4f} | {r['test_macro_f1']:.4f} | {r['parameters']:,} | {r['inference_ms']:.2f} |")
    text += [
        "", f"Selected deployment model: **{selected}**, chosen by validation macro F1 "
        "(validation accuracy, then parameter count break ties) before computing test metrics.", "",
        "Test accuracy, macro precision, macro recall, macro F1, per-class results and confusion "
        "matrices are stored in the results files. Latency is the mean of ten warmed single-image "
        "forward passes on the recorded runtime; it is not an end-to-end web latency measurement.", "",
        "![Model comparison](model_comparison.png)", "",
        "![Class distribution](class_distribution.png)", "",
        "## Deployment", "",
        "Run `streamlit run app.py` from the project folder. The application reads "
        "artifacts/deployment/model.keras and its metadata, and uses the same RGB and resize code "
        "as training. It shows all four softmax scores; these are not calibrated clinical probabilities.", "",
        "## Limitations", "",
        "This is an image classification educational prototype, not a diagnostic or triage system. "
        "Patient IDs are not supplied, so patient-level independence cannot be verified. "
        "The cleaning does not detect every near duplicate. Results on this small curated dataset "
        "do not establish performance on other hospitals, scanners, populations, MRI sequences, "
        "or non-MRI uploads. A no_tumor prediction is not evidence that a person is tumor-free. "
        "Independent patient-level external validation, probability calibration, and clinical "
        "review would be needed before any medical use.", "",
        "## Reproducibility", "",
        "See run_config.json, environment.json, class_names.json, the retained-image manifest, "
        "training history CSVs, and test_predictions.csv. A fixed seed improves repeatability; "
        "hardware and library differences can still change training results.", "",
        "## Dataset attribution", "",
        "The supplied README identifies Labeled MRI Brain Tumor Dataset Version 1 by the Roboflow "
        "user Ali Rostami and CC BY 4.0. Original page: "
        "https://universe.roboflow.com/ali-rostami/labeled-mri-brain-tumor-dataset . "
        "The provider's labeling statements have not been independently verified.", "",
    ]
    (output / "REPORT.md").write_text("\n".join(text), encoding="utf-8")
    df = pd.DataFrame(comparisons)
    df.to_csv(output / "model_comparison.csv", index=False)
    import matplotlib.pyplot as plt
    ax = df.set_index("model")[["validation_macro_f1", "test_macro_f1"]].plot.bar(figsize=(9, 5), ylim=(0, 1))
    ax.set(xlabel="Model", ylabel="Macro F1", title="Validation and held-out test comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output / "model_comparison.png", dpi=170)
    plt.close()


def train(args):
    global tf
    import tensorflow as tensorflow
    tf = tensorflow
    tf.keras.utils.set_random_seed(args.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    root = discover_dataset_root(args.data)
    output = Path(args.output).resolve()
    # Prevent old results being mixed into a new submission.
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}. Use a new --output directory.")
    models_dir, results_dir = output / "models", output / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    scanned, unreadable = scan_dataset(root)
    records, duplicates = remove_leakage(scanned)
    excluded = unreadable + duplicates
    manifest_fields = ["relative_path", "split", "class_name", "label", "width", "height", "pixel_hash", "source_family"]
    write_csv(results_dir / "data_manifest.csv", [{k:r[k] for k in manifest_fields} for r in records], manifest_fields)
    write_csv(results_dir / "excluded_images.csv", excluded, ["path", "reason"])
    summary = dict(readable_before_cleaning=len(scanned), unreadable_images=len(unreadable),
                   retained_images=len(records), excluded_images=len(excluded))
    write_json(results_dir / "dataset_summary.json", summary)
    write_json(output / "class_names.json", CLASS_NAMES)
    write_json(results_dir / "run_config.json", {**vars(args), "resolved_dataset_root": str(root)})
    import sklearn
    import PIL
    write_json(results_dir / "environment.json", dict(
        python=platform.python_version(), tensorflow=tf.__version__, numpy=np.__version__,
        pillow=PIL.__version__, sklearn=sklearn.__version__, platform=platform.platform(),
        gpu_devices=[d.name for d in tf.config.list_physical_devices("GPU")],
        generated_utc=datetime.now(timezone.utc).isoformat(),
    ))
    plots_for_data(records, results_dir)
    split_records = {s: [r for r in records if r["split"] == s] for s in ("train", "validation", "test")}
    print("Retained split counts:", {s:len(v) for s,v in split_records.items()})
    train_counts = Counter(r["label"] for r in split_records["train"])
    class_weights = {k:len(split_records["train"]) / (len(CLASS_NAMES) * train_counts[k]) for k in range(4)}
    write_json(results_dir / "class_weights.json", class_weights)
    datasets = {s:make_dataset(v, args.image_size, args.batch_size, s == "train", args.seed) for s,v in split_records.items()}
    validation_labels = np.array([r["label"] for r in split_records["validation"]])
    test_labels = np.array([r["label"] for r in split_records["test"]])
    candidates, histories = [], {}
    weights = "imagenet" if args.weights == "imagenet" else None
    if weights is None:
        print("WARNING: weights=none is a software test mode, not pretrained transfer learning.")
    for name in args.models:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.seed)
        print(f"\nTraining {name}")
        model, base = build_model(name, args.image_size, weights)
        compile_model(model, 1e-3)
        stage1_path = models_dir / f"{name}_stage1.keras"
        started = time.perf_counter()
        history = model.fit(datasets["train"], validation_data=datasets["validation"], epochs=args.epochs,
                            class_weight=class_weights, callbacks=callbacks(stage1_path), verbose=2)
        rows = [dict(epoch=i+1, phase="initial", **{k:float(v[i]) for k,v in history.history.items()})
                for i in range(len(history.history["loss"]))]
        model.load_weights(stage1_path)  # Best epoch, not the last epoch.
        first_loss = float(model.evaluate(datasets["validation"], verbose=0)[0])
        chosen_path = stage1_path
        if base is not None and args.fine_tune_epochs > 0:
            base.trainable = True
            cutoff = max(0, len(base.layers) - args.unfreeze_layers)
            for i, layer in enumerate(base.layers):
                layer.trainable = i >= cutoff and not isinstance(layer, tf.keras.layers.BatchNormalization)
            compile_model(model, 1e-5)
            stage2_path = models_dir / f"{name}_stage2.keras"
            fine = model.fit(datasets["train"], validation_data=datasets["validation"], epochs=args.fine_tune_epochs,
                             class_weight=class_weights, callbacks=callbacks(stage2_path), verbose=2)
            model.load_weights(stage2_path)
            fine_loss = float(model.evaluate(datasets["validation"], verbose=0)[0])
            if fine_loss < first_loss:
                chosen_path = stage2_path
            rows += [dict(epoch=len(rows)+i+1, phase="fine_tune", **{k:float(v[i]) for k,v in fine.history.items()})
                     for i in range(len(fine.history["loss"]))]
        model = tf.keras.models.load_model(chosen_path, compile=False)
        # Preserve the already validated native checkpoint without reserializing.
        shutil.copy2(chosen_path, models_dir / f"{name}.keras")
        model.save(models_dir / f"{name}.h5", include_optimizer=False)
        for temporary in models_dir.glob(f"{name}_stage*.keras"):
            temporary.unlink()
        probabilities = model.predict(datasets["validation"], verbose=0)
        metrics = calculate_metrics(validation_labels, probabilities)
        write_json(results_dir / f"{name}_validation_metrics.json", metrics)
        candidates.append(dict(model=name, validation_accuracy=metrics["accuracy"],
                               validation_macro_f1=metrics["macro_f1"], parameters=int(model.count_params()),
                               training_seconds=float(time.perf_counter()-started)))
        histories[name] = rows
    # Selection is fixed before reading any test prediction or test metric.
    selected = max(candidates, key=lambda r:(r["validation_macro_f1"], r["validation_accuracy"], -r["parameters"]))["model"]
    comparisons, prediction_rows = [], []
    for row in candidates:
        name = row["model"]
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(models_dir / f"{name}.keras", compile=False)
        probabilities = model.predict(datasets["test"], verbose=0)
        metrics = calculate_metrics(test_labels, probabilities)
        write_json(results_dir / f"{name}_test_metrics.json", metrics)
        write_csv(results_dir / f"{name}_confusion_matrix.csv", [
            dict(actual_class=c, **dict(zip(CLASS_NAMES, metrics["confusion_matrix"][i]))) for i,c in enumerate(CLASS_NAMES)
        ], ["actual_class"] + CLASS_NAMES)
        plot_model(histories[name], metrics["confusion_matrix"], name, results_dir)
        sample = load_image_array(split_records["test"][0]["path"], args.image_size)[None]
        for _ in range(3):
            model(sample, training=False).numpy()
        start = time.perf_counter()
        for _ in range(10):
            model(sample, training=False).numpy()
        latency = (time.perf_counter()-start) * 1000 / 10
        comparisons.append(dict(**row, test_accuracy=metrics["accuracy"],
                                test_macro_precision=metrics["macro_precision"],
                                test_macro_recall=metrics["macro_recall"], test_macro_f1=metrics["macro_f1"],
                                inference_ms=float(latency), selected=name == selected))
        for record, scores in zip(split_records["test"], probabilities):
            prediction_rows.append(dict(model=name, path=record["relative_path"], actual=record["class_name"],
                                        predicted=CLASS_NAMES[int(scores.argmax())],
                                        **{f"p_{c}":float(scores[i]) for i,c in enumerate(CLASS_NAMES)}))
    write_csv(results_dir / "test_predictions.csv", prediction_rows,
              ["model", "path", "actual", "predicted"] + [f"p_{c}" for c in CLASS_NAMES])
    write_json(results_dir / "model_comparison.json", comparisons)
    deployment = output / "deployment"
    deployment.mkdir(exist_ok=True)
    shutil.copy2(models_dir / f"{selected}.keras", deployment / "model.keras")
    write_json(deployment / "metadata.json", dict(
        model_name=selected, class_names=CLASS_NAMES, image_size=args.image_size,
        input_range="0..255; normalization is inside the saved model",
        resize="PIL bilinear with EXIF correction and RGB conversion",
        selection="highest validation macro F1; validation accuracy and parameter count break ties",
        pretrained_weights=args.weights, seed=args.seed, clinical_validation=False,
    ))
    generate_report(results_dir, comparisons, selected, summary)
    write_json(output / "COMPLETED.json", dict(selected_model=selected, models=args.models,
        pretrained_weights=args.weights, software_test_mode=args.weights == "none",
        retained_images=len(records), completed_utc=datetime.now(timezone.utc).isoformat()))
    print(f"\nFinished. Selected model: {selected}. Results: {results_dir}")
    print("Next: python make_submission.py --artifacts", output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="dataset", help="Extracted assigned dataset directory")
    parser.add_argument("--output", default="artifacts", help="New or empty output directory")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--unfreeze-layers", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet",
                        help="Use imagenet for the assignment; none is only for software tests")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32:
        parser.error("epochs and batch-size must be positive; image-size must be at least 32")
    if args.fine_tune_epochs < 0 or args.unfreeze_layers < 1:
        parser.error("fine-tune-epochs must be nonnegative and unfreeze-layers positive")
    if len(set(args.models)) != len(args.models):
        parser.error("Do not specify the same model twice")
    return args


if __name__ == "__main__":
    try:
        train(parse_args())
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}")
