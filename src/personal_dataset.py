"""Prepare Marcus's original M4A recordings without renaming or editing them."""

import argparse
import hashlib
from pathlib import Path
import re
import subprocess

import imageio_ffmpeg
import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
PERSONAL_CHORDS = ("A", "Am", "B", "Bm", "C", "D", "Dm", "E", "Em", "F", "G")
TECHNIQUES = ("downstrum", "repeated_strumming", "slow_picking")


def recording_number(path):
    """Read the number from a filename such as Nytt opptak 14.m4a."""
    match = re.fullmatch(r"Nytt opptak (\d+)", path.stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unexpected recording name: {path.name}")
    return int(match.group(1))


def ordered_recordings(source):
    """Use folders for chords and the reported recording order for playing styles."""
    rows = []
    for chord in PERSONAL_CHORDS:
        files = sorted((source / chord).glob("*.m4a"), key=recording_number)
        numbers = [recording_number(path) for path in files]
        if len(files) != 12 or len(set(numbers)) != 12:
            raise ValueError(f"Expected 12 uniquely numbered M4A files in {source / chord}")

        # Within each chord: four downstrums, four repeated strums, four picking takes.
        for index, path in enumerate(files):
            technique = TECHNIQUES[index // 4]
            if chord in ("B", "Bm", "Dm", "F") and index < 4:
                technique = "fast_strum"
            rows.append((path, chord, index + 1, technique))
    return rows


def prepare(source, output, metadata_path):
    """Make WAV copies and save their labels and audio details in a CSV table."""
    rows = []
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for original, chord, within_chord, technique in ordered_recordings(source):
        target = output / chord / (original.stem + ".wav")
        target.parent.mkdir(parents=True, exist_ok=True)

        # Make a WAV copy, keeping the original sample rate and channels.
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(original),
                        "-map", "0:a:0", "-vn", "-c:a", "pcm_f32le", str(target)],
                       check=True, capture_output=True)
        audio, sr = sf.read(target, always_2d=True)
        if not audio.size or not np.isfinite(audio).all():
            raise ValueError(f"Invalid decoded audio: {original}")

        # Save labels, audio details, and a checksum to detect source changes.
        rows.append({
            "filepath": target.relative_to(ROOT).as_posix(),
            "original_filepath": original.relative_to(ROOT).as_posix(),
            "original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
            "chord": chord, "source_dataset": "marcus_recordings",
            "guitar_id": "unknown", "player_id": "marcus",
            "original_split": "personal_collection",
            "recording_number": recording_number(original), "within_chord": within_chord,
            "technique": technique, "technique_basis": "user-reported order within chord",
            "label_basis": "chord folder", "picking_material": "unknown", "voicing": "unknown",
            "duration_s": len(audio) / sr, "sample_rate": sr, "channels": audio.shape[1],
            "peak_amplitude": float(np.max(np.abs(audio))),
        })

    # Write one metadata row per recording and print a collection overview.
    metadata = pd.DataFrame(rows)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_path, index=False)
    print(pd.crosstab(metadata.chord, metadata.technique).to_string())
    print(metadata.groupby(["sample_rate", "channels"]).size().to_string())
    print(metadata.duration_s.describe().to_string())
    print(f"Saved {len(metadata)} metadata rows to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/raw/data/marcus_recordings")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/marcus_recordings")
    parser.add_argument("--metadata", type=Path, default=ROOT / "reports/personal_11/metadata.csv")
    args = parser.parse_args()
    prepare(args.source.resolve(), args.output.resolve(), args.metadata.resolve())
