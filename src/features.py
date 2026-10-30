"""Prepare audio and describe each recording with 12 pitch-class values."""

from pathlib import Path

import librosa
import numpy as np

PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
DEFAULT_DURATION = 3.0
MIN_DURATION = 1.0


def preprocess_audio(
    filepath: str | Path,
    *,
    sr: int = 22050,
    duration: float = DEFAULT_DURATION,
    top_db: float = 30.0,
    min_duration: float = MIN_DURATION,
) -> tuple[np.ndarray, int, dict]:
    """Load mono audio, trim quiet edges, and keep up to `duration` seconds.

    Reject silence, invalid samples, and clips shorter than `min_duration`.
    Keep shorter valid clips without padding. Return the clip, sample rate,
    and crop times measured in the resampled recording.
    """
    # Check settings before loading the recording.
    if sr < 8000 or not np.isfinite(duration) or duration <= 0:
        raise ValueError("Use sr >= 8000 and a finite, positive duration.")
    if not np.isfinite(top_db) or top_db <= 0:
        raise ValueError("top_db must be finite and positive.")
    if not np.isfinite(min_duration) or not 0 < min_duration <= duration:
        raise ValueError("Require 0 < min_duration <= duration.")

    # Use one channel and a common sample rate for every recording.
    audio, sr = librosa.load(filepath, sr=sr, mono=True)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Empty or nonfinite recording: {filepath}")
    if np.max(np.abs(audio)) <= 1e-8:
        raise ValueError(f"Silent recording: {filepath}")

    # Find sound relative to the loudest frame; background sounds can trigger it too.
    _, bounds = librosa.effects.trim(
        audio, top_db=top_db, frame_length=2048, hop_length=256
    )

    # Keep the available sound, up to the chosen maximum length.
    start, active_end = map(int, bounds)
    length = round(duration * sr)
    stop = min(start + length, active_end)
    if stop - start < round(min_duration * sr):
        raise ValueError(f"Less than {min_duration:g}s after trimming: {filepath}")
    return audio[start:stop], sr, {
        "original_duration_s": len(audio) / sr,
        "start_s": start / sr,
        "stop_s": stop / sr,
        "trim_end_s": active_end / sr,
        "top_db": top_db,
        "selected_duration_s": (stop - start) / sr,
    }


def chromagram(audio: np.ndarray, sr: int = 22050, *, hop_length: int = 512) -> np.ndarray:
    """Measure the strength of 12 pitch classes over time, ignoring octaves.

    Scale each time frame by its strongest pitch class. Use fixed tuning
    because automatic estimates can vary on short clips. Six octaves from
    C2 cover guitar notes and harmonics without needing very low filters.
    """
    audio = np.asarray(audio)
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Expected a finite, nonempty mono waveform.")
    if np.max(np.abs(audio)) <= 1e-8:
        raise ValueError("Cannot extract chord features from silence.")
    return librosa.feature.chroma_cqt(
        y=audio, sr=sr, hop_length=hop_length, n_chroma=12,
        tuning=0.0, norm=np.inf, fmin=librosa.note_to_hz("C2"), n_octaves=6,
    )


def extract_chroma_features(filepath: str | Path, **preprocessing_options) -> np.ndarray:
    """Average pitch-class strength over time: 12 values in C, C#, ..., B order."""
    audio, sr, _ = preprocess_audio(filepath, **preprocessing_options)

    # Averaging gives the model 12 inputs even when clip lengths differ.
    return chromagram(audio, sr).mean(axis=1)
