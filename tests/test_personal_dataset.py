from pathlib import Path
import tempfile
import unittest

from src.personal_dataset import PERSONAL_CHORDS, ordered_recordings, recording_number


class PersonalDatasetTests(unittest.TestCase):
    def test_numeric_order_and_technique_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i, chord in enumerate(PERSONAL_CHORDS):
                (root / chord).mkdir()
                for n in reversed(range(87 + 12*i, 99 + 12*i)):
                    (root / chord / f"Nytt opptak {n}.m4a").touch()
            rows = ordered_recordings(root)
            self.assertEqual(len(rows), 12 * len(PERSONAL_CHORDS))
            self.assertEqual(recording_number(rows[12][0]), 99)
            self.assertEqual([r[3] for r in rows[:12]],
                             ["downstrum"]*4 + ["repeated_strumming"]*4 + ["slow_picking"]*4)
            rows[0][0].unlink()
            with self.assertRaises(ValueError):
                ordered_recordings(root)

    def test_invalid_name_rejected(self):
        with self.assertRaises(ValueError):
            recording_number(Path("extra_recording.m4a"))

    def test_numbers_are_only_relative_to_each_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for chord in PERSONAL_CHORDS:
                (root / chord).mkdir()
                for n in range(1, 25, 2):
                    (root / chord / f"Nytt opptak {n}.m4a").touch()
            rows = ordered_recordings(root)
            self.assertEqual(len(rows), 12 * len(PERSONAL_CHORDS))
            self.assertTrue(all(row[2] == index % 12 + 1 for index, row in enumerate(rows)))


if __name__ == "__main__":
    unittest.main()
