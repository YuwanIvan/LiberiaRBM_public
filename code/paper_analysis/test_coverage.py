"""Checks for scientifically material alignment failures, using saved diagnostics."""
import unittest
import pandas as pd
from .coverage import align_support
from .data import RELEASE


class CoverageAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = RELEASE / 'data/baseline_coverage'
        cls.training = pd.read_csv(source / 'state_support_summary.csv')
        cls.evaluation = pd.read_csv(source / 'evaluation_state_support_summary.csv')

    def test_reordered_inputs_do_not_change_coordinate_alignment(self):
        expected = align_support(self.training, self.evaluation)
        actual = align_support(self.training.sample(frac=1, random_state=7), self.evaluation.sample(frac=1, random_state=9))
        pd.testing.assert_frame_equal(expected, actual)

    def test_shifted_training_interval_rejected(self):
        shifted = self.evaluation.copy()
        mask = shifted.source.eq('discrete_review_deep_splitting')
        shifted.loc[mask, 'training_interval'] += 1
        with self.assertRaisesRegex(ValueError, 'misaligned'):
            align_support(self.training, shifted)

    def test_duplicate_coordinate_rejected(self):
        row = self.training[(self.training.source == 'training_sampler') & (self.training.variable == 'Q')].iloc[[0]]
        with self.assertRaisesRegex(ValueError, 'duplicate or incomplete'):
            align_support(pd.concat([self.training, row]), self.evaluation)

    def test_fine_grid_evaluation_cannot_substitute_for_e3_queries(self):
        evaluation = self.evaluation[self.evaluation.source == 'fine_grid_deep_splitting']
        with self.assertRaisesRegex(ValueError, 'duplicate or incomplete'):
            align_support(self.training, evaluation)


if __name__ == '__main__':
    unittest.main()
