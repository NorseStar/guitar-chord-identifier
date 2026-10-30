"""Inspect the locally downloaded isolated-guitar-chords dataset."""

from pathlib import Path

import pandas as pd
import soundfile as sf

from src.features import PITCH_CLASSES

CHORDS = tuple(root + quality for root in PITCH_CLASSES for quality in ("", "m"))


def inspect_dataset(root: str | Path) -> pd.DataFrame:
    """Build a file table from audio headers and chord folder names.

    Keep the original Train/Test groups. Include noise in the table, but
    mark it as not a chord. Reject folder names outside the known labels.
    """
    root = Path(root)
    rows = []

    # Read file information without loading the full audio.
    for split in ("Train", "Test"):
        if not (root / split).is_dir():
            raise FileNotFoundError(root / split)
        for path in sorted((root / split).rglob("*.wav")):
            label = path.parent.name
            if label not in CHORDS and label != "Noise":
                raise ValueError(f"Unsupported class folder: {path}")
            info = sf.info(path)
            rows.append({
                "filepath": path.relative_to(root).as_posix(),
                "chord": label if label in CHORDS else pd.NA,
                "source_label": label,
                "source_dataset": "severyn-k/isolated-guitar-chords",
                "guitar_id": "unknown", "player_id": "unknown",
                "original_split": split, "is_chord": label in CHORDS,
                "duration_s": info.duration, "sample_rate": info.samplerate,
                "channels": info.channels, "format": info.format,
                "subtype": info.subtype,
            })
    if not rows:
        raise ValueError(f"No WAV recordings found in {root}")
    return pd.DataFrame(rows)
