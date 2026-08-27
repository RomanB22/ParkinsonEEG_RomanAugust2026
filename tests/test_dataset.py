import unittest

from core.dataset import discover_recordings, load_participants, ordered_channel_inventory
from core.metadata import expected_channels_from_dataset


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

    def test_channel_inventory_returns_only_cohort_intersection(self):
        shared, union = ordered_channel_inventory(
            {
                "sub-001": ["Fp1", "Fz", "Cz"],
                "sub-002": ["Fz", "Cz", "Pz"],
                "sub-003": ["Cz", "Fz"],
            }
        )
        self.assertEqual(shared, ["Fz", "Cz"])
        self.assertEqual(union, ["Fp1", "Fz", "Cz", "Pz"])

    def test_channel_inventory_rejects_an_empty_intersection(self):
        with self.assertRaisesRegex(ValueError, "no EEG electrodes in common"):
            ordered_channel_inventory(
                {"sub-001": ["Fp1"], "sub-002": ["Pz"]}
            )


if __name__ == "__main__":
    unittest.main()
