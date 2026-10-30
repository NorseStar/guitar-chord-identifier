"""Predict a chord from a file or a prepared audio window."""

import argparse
import json
from pathlib import Path

import joblib
import librosa
import numpy as np

from src.features import MIN_DURATION, PITCH_CLASSES, chromagram, preprocess_audio

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models/chord_rf.joblib"


def load_model(path=DEFAULT_MODEL):
    """Load a trusted local model and check its pitch-class feature format."""
    # Only open trusted model files: joblib can execute code while loading.
    bundle = joblib.load(path)
    if bundle.get("feature_version") != 1 or bundle["feature_names"] != list(PITCH_CLASSES):
        raise ValueError("Model feature schema does not match this extractor.")
    return bundle


def rms_dbfs(audio):
    """Measure average volume in dBFS; more negative values mean quieter audio."""
    rms = np.sqrt(np.mean(np.square(np.asarray(audio, dtype=np.float64))))
    return float(20 * np.log10(max(float(rms), 1e-12)))


def unknown(reason):
    """Return a consistent result when the audio cannot be classified."""
    return {"prediction": "Unknown", "scores": {}, "reason": reason}


def predict_window(audio, sr, bundle, *, require_full=True):
    """Predict from mono audio at the model's sample rate.

    Require a full window by default; file crops may be shorter. Keep the
    window as supplied, without trimming. Scores are model estimates,
    not a measured chance of being correct.
    """
    # Check the audio shape, sample rate, and supported length.
    audio = np.asarray(audio, dtype=np.float32)
    settings = bundle["preprocessing"]
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        return unknown("Expected finite mono audio.")
    if sr != settings["sr"]:
        raise ValueError("Resample the window to the model sample rate first.")
    if require_full and len(audio) != round(settings["duration"] * sr):
        return unknown(f"Need a complete {settings['duration']:g}-second window.")
    if not require_full and not (round(settings.get("min_duration", MIN_DURATION) * sr)
                                <= len(audio) <= round(settings["duration"] * sr)):
        return unknown("Audio length outside the supported file-crop range.")

    # Reject very quiet audio before asking the model for a chord.
    level = rms_dbfs(audio)
    if level < bundle["min_rms_dbfs"]:
        return {"prediction": "No chord", "scores": {}, "rms_dbfs": level,
                "reason": "Signal below the quiet-audio threshold."}

    # Use the same 12 features as training, then rank chords by model score.
    features = chromagram(audio, sr).mean(axis=1)
    model = bundle["model"]
    probabilities = model.predict_proba(features.reshape(1, -1))[0]
    scores = dict(sorted(zip(model.classes_.tolist(), probabilities.tolist()),
                         key=lambda item: item[1], reverse=True))
    return {"prediction": next(iter(scores)), "scores": scores, "rms_dbfs": level,
            "reason": "Uncalibrated model scores; unfamiliar audio may still score highly."}


def predict_file(filepath, bundle):
    """Apply the training crop to a file, then predict using the same extractor."""
    settings = bundle["preprocessing"]
    audio, sr = librosa.load(filepath, sr=settings["sr"], mono=True)
    if not audio.size or not np.isfinite(audio).all():
        return unknown("Empty or nonfinite recording.")

    # Check the loudest part before trimming so leading silence does not hide a chord.
    peak_rms = float(librosa.feature.rms(y=audio).max())
    if 20 * np.log10(max(peak_rms, 1e-12)) < bundle["min_rms_dbfs"]:
        return {"prediction": "No chord", "scores": {}, "reason": "Signal below the quiet-audio threshold."}

    # Apply the saved training settings, then classify the selected clip.
    try:
        selected, sr, crop = preprocess_audio(filepath, **settings)
    except ValueError as exc:
        return unknown(str(exc))
    result = predict_window(selected, sr, bundle, require_full=False)
    result["crop"] = crop
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(json.dumps(predict_file(args.audio, load_model(args.model)), indent=2))
