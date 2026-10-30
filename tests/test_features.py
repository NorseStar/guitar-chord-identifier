"""Small checks for audio boundaries, resampling, and pitch-class ordering."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from src.features import PITCH_CLASSES, chromagram, preprocess_audio


class FeatureTests(unittest.TestCase):
    def test_leading_silence_and_stereo_resampling(self):
        sr = 44100
        t = np.arange(4 * sr) / sr
        tone = 0.3 * np.sin(2 * np.pi * 220 * t)
        audio = np.concatenate([np.zeros(sr), tone])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            sf.write(path, np.column_stack([audio, audio]), sr)
            selected, output_sr, info = preprocess_audio(path)
            shorter, _, _ = preprocess_audio(path, duration=2.0)
        self.assertEqual(output_sr, 22050)
        self.assertEqual(selected.shape, (66150,))
        self.assertEqual(shorter.shape, (44100,))
        self.assertAlmostEqual(info["start_s"], 1.0, delta=0.06)
        self.assertTrue(np.isfinite(selected).all())

    def test_silence_and_short_recordings_rejected(self):
        sr = 22050
        t = np.arange(sr // 2) / sr
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.wav"
            for audio in (np.zeros(sr * 3), 0.3 * np.sin(2 * np.pi * 220 * t)):
                sf.write(path, audio, sr)
                with self.assertRaises(ValueError):
                    preprocess_audio(path)

    def test_major_minor_third_is_visible(self):
        sr = 22050
        t = np.arange(sr * 2) / sr
        for notes, third, other in (([57, 61, 64], "C#", "C"), ([57, 60, 64], "C", "C#")):
            audio = sum(np.sin(2 * np.pi * (440 * 2 ** ((note - 69) / 12)) * t)
                        for note in notes) / 3
            vector = chromagram(audio, sr).mean(axis=1)
            strongest = set(np.argsort(vector)[-3:])
            self.assertEqual(strongest, {PITCH_CLASSES.index(p) for p in ("A", third, "E")})
            self.assertGreater(vector[PITCH_CLASSES.index(third)], vector[PITCH_CLASSES.index(other)])

    def test_short_valid_audio_is_not_padded_or_rejected(self):
        sr = 22050
        t = np.arange(2 * sr) / sr
        tone = 0.3 * np.sin(2 * np.pi * 220 * t)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short.wav"
            sf.write(path, tone, sr)
            original, _ = sf.read(path, dtype="float32")
            selected, _, info = preprocess_audio(path)
        np.testing.assert_array_equal(selected, original)
        self.assertEqual(info["selected_duration_s"], 2.0)
        self.assertEqual(chromagram(selected, sr).mean(axis=1).shape, (12,))

    def test_invalid_waveforms_rejected(self):
        for audio in (np.array([]), np.array([np.nan]), np.zeros(100), np.ones((2, 100))):
            with self.assertRaises(ValueError):
                chromagram(audio)


if __name__ == "__main__":
    unittest.main()
