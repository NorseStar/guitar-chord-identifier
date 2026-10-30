"""Train the first baseline: python -m src.train (from the project root)."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from src.dataset import CHORDS, inspect_dataset
from src.features import DEFAULT_DURATION, MIN_DURATION, PITCH_CLASSES, extract_chroma_features

ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING = {"sr": 22050, "duration": DEFAULT_DURATION, "top_db": 30.0,
                 "min_duration": MIN_DURATION}


def training_rows(metadata):
    """Keep chord recordings from Train; leave Test and noise out."""
    return metadata.loc[(metadata.original_split == "Train") & metadata.is_chord].copy()


def split_rows(labels, seed=42):
    """Reserve 20% for validation, keeping the chord proportions similar."""
    return train_test_split(np.arange(len(labels)), test_size=0.2,
                            random_state=seed, stratify=labels)


def reuse_split(manifest, previous):
    """Reuse saved train/validation groups and check that the audio is unchanged."""
    saved = previous.set_index("filepath")
    if not saved.index.is_unique:
        raise ValueError("Saved split contains duplicate file paths.")
    saved = saved.reindex(manifest.filepath)
    if saved.partition.isna().any() or not saved.partition.isin(["train", "validation"]).all():
        raise ValueError("Every usable recording needs a valid saved split assignment.")
    if not np.array_equal(saved.sha256.to_numpy(), manifest.sha256.to_numpy()):
        raise ValueError("Audio files differ from the saved split; check the dataset.")
    partitions = saved.partition.to_numpy()
    return np.flatnonzero(partitions == "train"), np.flatnonzero(partitions == "validation")


def train(data_root, output, model_path, split_manifest=None):
    """Extract features, train the model, and save validation results."""
    # 1. Read the training files and turn each recording into 12 features.
    output.mkdir(parents=True, exist_ok=True)
    metadata = training_rows(inspect_dataset(data_root))
    rows, vectors, errors = [], [], []
    for row in metadata.itertuples():
        path = data_root / row.filepath
        try:
            vector = extract_chroma_features(path, **PREPROCESSING)
            if vector.shape != (12,) or not np.isfinite(vector).all():
                raise ValueError("Invalid chroma vector")
        except (ValueError, RuntimeError, OSError) as exc:
            errors.append({"filepath": row.filepath, "error": str(exc).replace(str(data_root), "<data>")})
            continue
        vectors.append(vector)
        rows.append({"filepath": row.filepath, "chord": row.chord,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    # 2. Record failed files and check for missing chords or duplicate audio.
    pd.DataFrame(errors, columns=["filepath", "error"]).to_csv(output / "extraction_errors.csv", index=False)
    manifest = pd.DataFrame(rows)
    if manifest.empty or set(manifest.chord) != set(CHORDS):
        raise ValueError("Expected usable training recordings for all 24 chords.")
    if manifest.sha256.duplicated().any():
        raise ValueError("Duplicate training files found; resolve them before splitting.")

    # X holds the pitch features; y holds the correct chord for each recording.
    X, y = np.asarray(vectors), manifest.chord.to_numpy()

    # 3. Separate training and validation recordings before fitting the model.
    if split_manifest is None:
        fit, validation = split_rows(y)
    else:
        fit, validation = reuse_split(manifest, pd.read_csv(split_manifest))
    if not len(validation) or set(y[fit]) != set(CHORDS):
        raise ValueError("Need all chords in training and a nonempty validation partition.")

    # 4. Fit on training recordings, then predict the held-out validation set.
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=42, n_jobs=-1)
    model.fit(X[fit], y[fit])
    predicted = model.predict(X[validation])

    # Compare with a simple model that always guesses the most common chord.
    dummy = DummyClassifier(strategy="most_frequent").fit(X[fit], y[fit])
    report = classification_report(y[validation], predicted, labels=list(CHORDS),
                                   output_dict=True, zero_division=0)
    metrics = {
        "evaluation_partition": "validation drawn only from original Train",
        "seed": 42, "validation_fraction": 0.2,
        "training_rows": len(fit), "validation_rows": len(validation),
        "excluded_rows": len(errors),
        "duration_s": PREPROCESSING["duration"],
        "split_strategy": "saved assignments" if split_manifest else "stratified random split",
        "accuracy": accuracy_score(y[validation], predicted),
        "dummy_accuracy": accuracy_score(y[validation], dummy.predict(X[validation])),
        "classification_report": report,
    }

    # 5. Save scores and file-level predictions so mistakes can be inspected.
    (output / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    manifest["partition"] = "train"
    manifest.loc[validation, "partition"] = "validation"
    manifest.to_csv(output / "split_manifest.csv", index=False)
    predictions = manifest.iloc[validation].copy()
    predictions["prediction"] = predicted
    predictions.to_csv(output / "validation_predictions.csv", index=False)

    # The confusion matrix shows which chords are mistaken for each other.
    fig, ax = plt.subplots(figsize=(12, 11))
    ConfusionMatrixDisplay.from_predictions(y[validation], predicted, labels=list(CHORDS),
                                           ax=ax, colorbar=False, xticks_rotation=90)
    ax.set_title("Random Forest — validation recordings only")
    fig.tight_layout()
    fig.savefig(output / "validation_confusion_matrix.png", dpi=140)
    plt.close(fig)

    # 6. Save the model together with the settings prediction must reuse.
    bundle = {
        "model": model, "preprocessing": PREPROCESSING,
        "feature_names": list(PITCH_CLASSES), "feature_version": 1,
        "feature_description": "Mean chroma_cqt; C2, six octaves, hop=512, tuning=0, norm=inf",
        "min_rms_dbfs": -50.0,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "librosa", "scikit-learn", "joblib", "soundfile")},
        "training_rows": len(fit), "seed": 42,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)
    (output / "model_config.json").write_text(
        json.dumps({key: value for key, value in bundle.items() if key != "model"}, indent=2),
        encoding="utf-8")
    print(f"Train: {len(fit)}; validation: {len(validation)}; excluded: {len(errors)}")
    print(f"Validation accuracy: {metrics['accuracy']:.3f}; macro F1: {report['macro avg']['f1-score']:.3f}")
    print(f"Most-frequent baseline accuracy: {metrics['dummy_accuracy']:.3f}")
    print(f"Saved {model_path}. Original Test recordings were not used for features or fitting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/data")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/baseline")
    parser.add_argument("--model", type=Path, default=ROOT / "models/chord_rf.joblib")
    parser.add_argument("--split-manifest", type=Path,
                        default=ROOT / "reports/training_split.csv",
                        help="Reuse the original baseline assignments, even if some files are now too short.")
    args = parser.parse_args()
    train(args.data, args.output, args.model, args.split_manifest)
