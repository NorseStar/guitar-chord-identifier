import unittest
from unittest.mock import patch
import numpy as np
from src.streaming import stream_chunk, fresh_state

class Model:
    classes_ = np.array(["E", "Em"])

class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.bundle = {"model": Model(), "preprocessing": {"sr": 22050, "duration": 3.0,
                       "min_duration": 1.0}, "min_rms_dbfs": -50.0}

    def test_buffer_bounded_and_silence_clears_old_chord(self):
        state = None
        prediction = {"prediction": "E", "scores": {"E": 0.8, "Em": 0.2}}
        with patch("src.streaming.predict_window", return_value=prediction):
            for _ in range(80):
                state, chord, scores, graph = stream_chunk(
                    (22050, np.ones(11025, dtype=np.float32) * 0.1), state, self.bundle)
        self.assertEqual(chord, "E")
        self.assertEqual(len(state["audio"]), 66150)
        self.assertLessEqual(len(graph), 122)
        state, chord, scores, graph = stream_chunk((22050, np.zeros(11025)), state, self.bundle)
        self.assertIn("No chord", chord)
        self.assertEqual(len(state["audio"]), 0)
        self.assertEqual(scores, {})
        self.assertIsNone(state["scores"])

    def test_stereo_integer_input_and_session_isolation(self):
        a = fresh_state()
        b = fresh_state()
        a, chord, _, _ = stream_chunk((48000, np.full((24000, 2), 3200, dtype=np.int16)), a, self.bundle)
        self.assertEqual(chord, "Listening…")
        self.assertEqual(len(a["audio"]), 11025)
        self.assertEqual(len(b["audio"]), 0)
        self.assertLess(float(a["audio"].max()), 0.2)

    def test_invalid_chunk_resets(self):
        state, chord, scores, graph = stream_chunk((22050, np.array([np.nan])), None, self.bundle)
        self.assertIn("Unknown", chord)
        self.assertEqual(scores, {})

if __name__ == "__main__":
    unittest.main()
