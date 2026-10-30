from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd
import soundfile as sf

from src.features import extract_chroma_features
from src.predict import predict_file, predict_window
from src.train import PREPROCESSING, reuse_split, split_rows, training_rows


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.model = Mock()
        self.model.classes_ = np.array(["A", "Am"])
        self.model.predict_proba.return_value = np.array([[0.2, 0.8]])
        self.bundle = {"model": self.model, "preprocessing": PREPROCESSING,
                       "min_rms_dbfs": -50.0}

    def test_silent_window_never_calls_classifier(self):
        result = predict_window(np.zeros(66150), 22050, self.bundle)
        self.assertEqual(result["prediction"], "No chord")
        self.assertEqual(result["scores"], {})
        self.model.predict_proba.assert_not_called()

    def test_incomplete_and_invalid_windows_abstain(self):
        for audio in (np.ones(100), np.ones(44100), np.full(66150, np.nan)):
            self.assertEqual(predict_window(audio, 22050, self.bundle)["prediction"], "Unknown")
        self.model.predict_proba.assert_not_called()

    def test_file_inference_matches_training_features(self):
        sr = 44100
        t = np.arange(4 * sr) / sr
        tone = 0.2 * np.sin(2 * np.pi * 220 * t)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chord.wav"
            sf.write(path, np.concatenate([np.zeros(sr), tone]), sr)
            result = predict_file(path, self.bundle)
            expected = extract_chroma_features(path, **PREPROCESSING)
        actual = self.model.predict_proba.call_args.args[0][0]
        np.testing.assert_allclose(actual, expected)
        self.assertEqual(result["prediction"], "Am")
        self.assertEqual(result["scores"], {"Am": 0.8, "A": 0.2})

    def test_file_silence_is_no_chord(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "silence.wav"
            sf.write(path, np.zeros(44100), 22050)
            self.assertEqual(predict_file(path, self.bundle)["prediction"], "No chord")
        self.model.predict_proba.assert_not_called()

    def test_short_file_predicts_with_same_features_as_training(self):
        sr = 22050
        t = np.arange(2 * sr) / sr
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short.wav"
            sf.write(path, 0.2 * np.sin(2 * np.pi * 220 * t), sr)
            result = predict_file(path, self.bundle)
            expected = extract_chroma_features(path, **PREPROCESSING)
        self.assertEqual(result["prediction"], "Am")
        self.assertEqual(result["crop"]["selected_duration_s"], 2.0)
        np.testing.assert_allclose(self.model.predict_proba.call_args.args[0][0], expected)

    def test_original_test_and_noise_excluded(self):
        metadata = pd.DataFrame({"original_split": ["Train", "Test", "Train"],
                                 "is_chord": [True, True, False]})
        self.assertEqual(training_rows(metadata).index.tolist(), [0])

    def test_split_is_repeatable_disjoint_and_stratified(self):
        labels = np.repeat(["A", "Am", "C"], 25)
        train, validation = split_rows(labels)
        again_train, again_validation = split_rows(labels)
        np.testing.assert_array_equal(train, again_train)
        np.testing.assert_array_equal(validation, again_validation)
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(set(train) | set(validation), set(range(len(labels))))
        self.assertEqual(set(labels[train]), set(labels[validation]))

    def test_saved_split_survives_rejections_and_reordering(self):
        old = pd.DataFrame({"filepath": ["a", "b", "c"], "sha256": ["aa", "bb", "cc"],
                            "partition": ["train", "validation", "train"]})
        current = old.iloc[[2, 1]].drop(columns="partition").reset_index(drop=True)
        fit, validation = reuse_split(current, old)
        np.testing.assert_array_equal(fit, [0])
        np.testing.assert_array_equal(validation, [1])
        current.loc[0, "sha256"] = "changed"
        with self.assertRaises(ValueError):
            reuse_split(current, old)


if __name__ == "__main__":
    unittest.main()
