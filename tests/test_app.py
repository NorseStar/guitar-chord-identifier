import unittest
from unittest.mock import patch

from app import classify_recording


class AppTests(unittest.TestCase):
    def test_empty_input_clears_scores(self):
        chord, scores, message = classify_recording(None, {})
        self.assertEqual(scores, {})
        self.assertIn("upload", message)

    @patch("app.predict_file")
    def test_top_three_are_not_renormalised(self, predict):
        predict.return_value = {"prediction": "A", "scores": {"A": .5, "Am": .2, "E": .1, "C": .1},
                                "crop": {"start_s": 1, "stop_s": 3.5}}
        chord, scores, message = classify_recording("audio.wav", {})
        self.assertEqual(chord, "A")
        self.assertEqual(scores, {"A": .5, "Am": .2, "E": .1})
        self.assertIn("2.50", message)

    @patch("app.predict_file")
    def test_quiet_or_short_recordings_show_no_class_scores(self, predict):
        for state in ("No chord", "Unknown"):
            predict.return_value = {"prediction": state, "scores": {}}
            _, scores, message = classify_recording("audio.wav", {})
            self.assertEqual(scores, {})
            self.assertTrue(message)


if __name__ == "__main__":
    unittest.main()
