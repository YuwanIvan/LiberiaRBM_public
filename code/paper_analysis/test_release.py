"""Regression guards for provenance and scientific row selection."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .source_values import table_value
from .validation import validate_inputs
import pandas as pd
import hashlib


class ReleaseGuards(unittest.TestCase):
    def test_seed16_selection_ignores_other_training_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/'data/sources/baseline_policy_seed_rows.csv'
            source.parent.mkdir(parents=True)
            source.write_text('stream_id,policy,combined_objective\n17,E3_mean_rollout,999\n16,E3_mean_rollout,-12.5\n16,Static,777\n')
            cell={'source_relative':'baseline_policy_seed_rows.csv','source_field_or_formula':'combined_objective',
                  'row':'Our policy','row_filter':'stream_id=16; policy=E3_mean_rollout'}
            with patch('paper_analysis.source_values.RELEASE',root):
                self.assertEqual(table_value(cell),-12.5)
                with source.open('a') as handle:
                    handle.write('16,E3_mean_rollout,-13\n')
                with self.assertRaises(ValueError):
                    table_value(cell)

    def test_modified_frozen_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input.csv'
            source.write_bytes(b'original')
            manifest = pd.DataFrame([{
                'bundle_path': 'input.csv',
                'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            }])
            source.write_bytes(b'modified')
            with patch('paper_analysis.validation.RELEASE', root), \
                    patch('paper_analysis.validation.read', return_value=manifest):
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    validate_inputs()


if __name__=='__main__':
    unittest.main()
