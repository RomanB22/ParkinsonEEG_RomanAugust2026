import unittest

from src.dataset import discover_recordings, load_participants
from src.metadata import expected_channels_from_dataset


class DatasetTests(unittest.TestCase):
    def test_participants_and_recordings_match(self):
        participants = load_participants("dataset")
        recordings = discover_recordings("dataset", "Rest")
        self.assertEqual(len(participants), 149)
        self.assertEqual(len(recordings), 149)
        self.assertEqual(participants["GROUP"].value_counts().to_dict(), {"PD": 100, "Control": 49})

    def test_expected_channels_excludes_auxiliary(self):
        channels = expected_channels_from_dataset("dataset", "Rest", ["Resp", "X", "Y", "Z"])
        self.assertNotIn("Resp", channels)
        self.assertNotIn("X", channels)
        self.assertIn("Fp1", channels)


if __name__ == "__main__":
    unittest.main()

