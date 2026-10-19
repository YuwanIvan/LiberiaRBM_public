import unittest

import numpy as np

import state_support_alignment as alignment


class StateSupportAlignmentTests(unittest.TestCase):
    def envelope(self, intervals):
        result = {}
        for interval in range(intervals):
            for variable in ("Q", "Z_0", "Z_1", "aggregate_CHA_inventory", "minimum_CHA_inventory", "maximum_CHA_inventory"):
                result[(interval, variable)] = {
                    "min": -100.0,
                    "p01": -50.0,
                    "p05": -10.0,
                    "p99": 50.0,
                    "max": 100.0,
                }
        return result

    def test_review_mapping_for_m110_and_m1100(self):
        self.assertEqual(alignment.fine_intervals_for_review(3, 5), (15, 16, 17, 18, 19))
        self.assertEqual(
            alignment.fine_intervals_for_review(3, 50),
            tuple(range(150, 200)),
        )

    def test_periodic_virtual_states_map_to_consuming_networks(self):
        # Two review periods and three fine networks per review period.
        virtual_q = np.asarray(
            [
                [[10.0, 11.0], [9.0, 10.0], [8.0, 9.0], [7.0, 8.0]],
                [[7.0, 8.0], [6.0, 7.0], [5.0, 6.0], [4.0, 5.0]],
            ]
        )
        virtual_z = np.zeros((2, 4, 2, 2), dtype=float)
        for review in range(2):
            for offset in range(4):
                virtual_z[review, offset, :, 0] = 100 * review + 10 * offset
                virtual_z[review, offset, :, 1] = 1 + review + offset
        result = {
            "z_hist": np.zeros((3, 2, 2), dtype=float),
            "i_hist": np.zeros((3, 2), dtype=float),
            "virtual_q_hist": virtual_q,
            "virtual_z_hist": virtual_z,
        }
        rows = alignment.periodic_state_support_rows(
            result,
            num_fine_intervals=6,
            source="periodic",
            envelope=self.envelope(6),
        )
        q_rows = [row for row in rows if row["variable"] == "Q"]
        self.assertEqual([row["interval"] for row in q_rows], list(range(6)))
        self.assertEqual([row["review_interval"] for row in q_rows], [0, 0, 0, 1, 1, 1])
        self.assertEqual([row["fine_offset"] for row in q_rows], [0, 1, 2, 0, 1, 2])
        self.assertEqual(q_rows[4]["p50"], 6.5)
        self.assertEqual(
            {row["alignment_mode"] for row in q_rows},
            {"periodic_virtual_state_exact_fine_interval"},
        )

    def test_periodic_start_state_fallback_still_covers_all_fine_intervals(self):
        result = {
            "z_hist": np.zeros((3, 2, 2), dtype=float),
            "i_hist": np.asarray([[10.0, 10.0], [5.0, 5.0], [0.0, 0.0]]),
        }
        rows = alignment.periodic_state_support_rows(
            result,
            num_fine_intervals=10,
            source="periodic",
            envelope=self.envelope(10),
        )
        q_rows = [row for row in rows if row["variable"] == "Q"]
        self.assertEqual([row["interval"] for row in q_rows], list(range(10)))
        self.assertEqual([row["p50"] for row in q_rows[:5]], [10.0] * 5)
        self.assertEqual([row["p50"] for row in q_rows[5:]], [5.0] * 5)

    def test_fine_grid_mapping_is_one_to_one(self):
        result = {
            "z_hist": np.zeros((5, 2, 2), dtype=float),
            "i_hist": np.arange(10, dtype=float).reshape(5, 2),
        }
        rows = alignment.fine_grid_state_support_rows(
            result,
            source="fine",
            envelope=self.envelope(4),
        )
        q_rows = [row for row in rows if row["variable"] == "Q"]
        self.assertEqual([row["interval"] for row in q_rows], [0, 1, 2, 3])
        self.assertEqual(
            {row["alignment_schema_version"] for row in q_rows},
            {alignment.ALIGNMENT_SCHEMA_VERSION},
        )

    def test_support_distance_reports_magnitude_not_only_outside_fraction(self):
        result = {
            "z_hist": np.zeros((2, 2, 2), dtype=float),
            "i_hist": np.asarray([[0.0, 150.0], [0.0, 0.0]]),
        }
        rows = alignment.fine_grid_state_support_rows(
            result,
            source="fine",
            envelope=self.envelope(1),
        )
        q_row = next(row for row in rows if row["variable"] == "Q")
        self.assertEqual(q_row["outside_training_p01_p99_fraction"], 0.5)
        self.assertGreater(q_row["mean_normalized_distance_outside_training_p01_p99"], 0.0)
        self.assertGreater(q_row["max_normalized_distance_outside_training_min_max"], 0.0)


if __name__ == "__main__":
    unittest.main()
