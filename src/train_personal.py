"""Train the personal model: python -m src.train_personal."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.features import extract_chroma_features, PITCH_CLASSES

ROOT = Path(__file__).resolve().parents[1]

def train():
    output = ROOT / "reports/personal_11"
    metadata = pd.read_csv(output / "metadata.csv")
    settings = {"sr": 22050, "duration": 3.0, "top_db": 30.0, "min_duration": 1.0}
    vectors = []
    for filepath in metadata.filepath:
        vectors.append(extract_chroma_features(ROOT / filepath, **settings))
    X = np.array(vectors)
    y = metadata.chord.to_numpy()
    groups = metadata.chord + "_" + metadata.technique
    train_rows, validation_rows = train_test_split(
        np.arange(len(y)), test_size=0.25, random_state=42, stratify=groups)
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X[train_rows], y[train_rows])
    predictions = model.predict(X[validation_rows])
    metrics = classification_report(y[validation_rows], predictions, output_dict=True, zero_division=0)
    metrics["training_recordings"] = len(train_rows)
    metrics["validation_recordings"] = len(validation_rows)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    split = metadata.copy()
    split["split"] = "training"
    split.loc[validation_rows, "split"] = "validation"
    split.to_csv(output / "split.csv", index=False)
    results = metadata.iloc[validation_rows][["filepath", "chord", "technique"]].copy()
    results["prediction"] = predictions
    results.to_csv(output / "validation_predictions.csv", index=False)
    labels = model.classes_
    pd.DataFrame(confusion_matrix(y[validation_rows], predictions, labels=labels),
                 index=labels, columns=labels).to_csv(output / "confusion_matrix.csv")
    print("Validation accuracy:", accuracy_score(y[validation_rows], predictions))
    print(classification_report(y[validation_rows], predictions, zero_division=0))
    # Validation is finished. Fit a separate app model using the whole collection.
    final_model = RandomForestClassifier(n_estimators=300, random_state=42)
    final_model.fit(X, y)
    bundle = {"model": final_model, "preprocessing": settings,
              "feature_names": list(PITCH_CLASSES), "feature_version": 1,
              "min_rms_dbfs": -50.0, "training_rows": len(y), "seed": 42}
    destination = ROOT / "models/my_guitar_rf.joblib"
    destination.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, destination)
    print("Saved", destination)

if __name__ == "__main__":
    train()
