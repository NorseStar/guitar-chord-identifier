"""Test the saved model on personal recordings without retraining it."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from src.personal_dataset import ROOT, PERSONAL_CHORDS
from src.predict import DEFAULT_MODEL, load_model, predict_file


def evaluate(metadata_path, model_path, output):
    """Predict each personal recording and save scores, tables, and a chart."""

    # 1. Load the recording list and the existing model.
    metadata = pd.read_csv(metadata_path)
    if not metadata.original_split.eq("external_evaluation").all():
        raise ValueError("Expected a separate external evaluation collection.")
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    bundle = load_model(model_path)
    rows = []

    # 2. Check each original file is unchanged, then predict its WAV copy.
    for row in metadata.itertuples():
        original = ROOT / row.original_filepath
        if hashlib.sha256(original.read_bytes()).hexdigest() != row.original_sha256:
            raise ValueError(f"Source file changed since preparation: {original}")
        try:
            result = predict_file(ROOT / row.filepath, bundle)
        except (OSError, ValueError, RuntimeError) as exc:
            result = {"prediction": "Error", "scores": {}, "reason": str(exc)}
        scores = result["scores"]
        crop = result.get("crop", {})
        rows.append({"original_filepath": row.original_filepath, "chord": row.chord,
                     "technique": row.technique, "prediction": result["prediction"],
                     "correct": result["prediction"] == row.chord,
                     "top3_correct": row.chord in list(scores)[:3],
                     "top_score": next(iter(scores.values()), None),
                     "expected_chord_score": scores.get(row.chord),
                     "start_s": crop.get("start_s"), "stop_s": crop.get("stop_s"),
                     "reason": result["reason"], "scores": json.dumps(scores)})

    # 3. Save individual predictions and accuracy by chord and playing style.
    results = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "predictions.csv", index=False)
    by_chord = results.groupby("chord").agg(recordings=("correct", "size"),
                                            correct=("correct", "sum"), accuracy=("correct", "mean"))
    by_technique = results.groupby("technique").agg(recordings=("correct", "size"),
                                                   correct=("correct", "sum"), accuracy=("correct", "mean"))
    by_chord.to_csv(output / "by_chord.csv")
    by_technique.to_csv(output / "by_technique.csv")

    # Count rejected or failed recordings too, so accuracy includes every attempt.
    metrics = {
        "recordings": len(results), "correct": int(results.correct.sum()),
        "accuracy": float(results.correct.mean()), "top3_accuracy": float(results.top3_correct.mean()),
        "abstentions_or_errors": int(results.prediction.isin(["No chord", "Unknown", "Error"]).sum()),
        "model_sha256": model_hash, "model_preprocessing": bundle["preprocessing"],
        "label_basis": "user-confirmed chord folders; techniques from recording order",
        "scope": "Seven chords from one personal collection, all 24 model outputs allowed; no retraining.",
        "classification_report": classification_report(results.chord, results.prediction,
            labels=list(PERSONAL_CHORDS), output_dict=True, zero_division=0),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # 4. Show recorded chords as rows and predicted chords as columns.
    # Include guesses outside the seven chords in the personal collection.
    columns = list(PERSONAL_CHORDS) + sorted(set(results.prediction) - set(PERSONAL_CHORDS))
    matrix = pd.crosstab(results.chord, results.prediction).reindex(index=PERSONAL_CHORDS, columns=columns, fill_value=0)
    matrix.to_csv(output / "confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(max(8, len(columns) * .7), 6))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=12)
    ax.set(xticks=np.arange(len(columns)), xticklabels=columns,
           yticks=np.arange(len(PERSONAL_CHORDS)), yticklabels=PERSONAL_CHORDS,
           xlabel="Predicted chord", ylabel="Recorded chord", title="Marcus's recordings — existing model, no retraining")
    for i in range(len(matrix)):
        for j in range(len(columns)):
            value = int(matrix.iloc[i, j])
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > 6 else "black")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=140)
    plt.close(fig)

    # Confirm that evaluation did not change the saved model file.
    assert model_hash == hashlib.sha256(model_path.read_bytes()).hexdigest()
    print(f"Accuracy: {metrics['correct']}/{len(results)} ({metrics['accuracy']:.1%})")
    print(by_chord.to_string())
    print(by_technique.to_string())
    print("Abstentions/errors:", metrics['abstentions_or_errors'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=ROOT / "reports/marcus_recordings/metadata.csv")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/marcus_recordings")
    args = parser.parse_args()
    evaluate(args.metadata, args.model, args.output)
