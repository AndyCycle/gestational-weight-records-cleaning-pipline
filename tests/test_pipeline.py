import copy
import json
import tempfile
import unittest
from pathlib import Path

from portable.pipeline import normalize, screen, apply_review, run

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'examples/config.json').read_text())
HEADERS = list(CONFIG['columns'].values())


def source(rid, day, weight, pid='0001', delivery=''):
    return dict(record_id=rid, pregnancy_id=pid, gestation_day=str(day),
                weight=str(weight), height='160', delivery_day=str(delivery))


class PipelineTests(unittest.TestCase):
    def normalize(self, rows, config=None):
        return normalize(rows, HEADERS, config or CONFIG)

    def test_jin_and_meter(self):
        config = copy.deepcopy(CONFIG)
        config['units'] = dict(weight='jin', height='m')
        row = source('r', 60, 100)
        row['height'] = '1.6'
        records = screen(self.normalize([row], config), config)
        self.assertEqual(records[0]['weight_raw'], 100)
        self.assertEqual(records[0]['weight'], 50)
        self.assertEqual(records[0]['BMI'], 19.53)
        self.assertEqual(records[0]['pregnancy_id'], '0001')

    def test_mixed_has_no_bmi_or_synthetic_day0(self):
        config = copy.deepcopy(CONFIG)
        config['units']['weight'] = 'mixed'
        rows = screen(self.normalize([source('r', 60, 110)], config), config)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['gestation_day'], 60)
        self.assertEqual(rows[0]['weight'], 110)
        self.assertIsNone(rows[0]['BMI'])

    def test_real_high_weight_is_preserved(self):
        rows = screen(self.normalize([source(str(i), i * 60, w) for i, w in enumerate([110, 112, 115])]), CONFIG)
        self.assertEqual([r['weight'] for r in rows], [110, 112, 115])
        self.assertTrue(all(not r['flags'] for r in rows))

    def test_local_factor_two_candidate_only(self):
        rows = screen(self.normalize([source(str(i), 60 + i * 60, w) for i, w in enumerate([50, 104, 54])]), CONFIG)
        self.assertIn('factor_two_candidate', rows[1]['flags'])
        self.assertEqual(rows[1]['weight'], 104)

    def test_duplicate_day_retained_and_not_used_as_neighbor(self):
        rows = screen(self.normalize([source('a', 60, 50), source('b', 60, 100)]), CONFIG)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all('duplicate_pregnancy_day' in r['flags'] for r in rows))

    def test_known_postpartum_and_missing_delivery(self):
        config = copy.deepcopy(CONFIG)
        config['rules']['exclude_postpartum'] = True
        rows = screen(self.normalize([source('a', 280, 65, delivery=280),
                                     source('b', 281, 55),
                                     source('c', 290, 60, pid='other')]), config)
        self.assertEqual([r['excluded'] for r in rows], [False, True, False])

    def test_conflicting_delivery_fails(self):
        with self.assertRaises(ValueError):
            screen(self.normalize([source('a', 60, 50, delivery=270), source('b', 90, 51, delivery=280)]), CONFIG)

    def test_review_recalculates_bmi(self):
        rows = self.normalize([source('a', 60, 100)])
        apply_review(rows, [dict(record_id='a', expected_weight='100', operation='set_weight', new_weight='50', reason='verified')])
        screen(rows, CONFIG)
        self.assertEqual(rows[0]['weight_raw'], 100)
        self.assertEqual(rows[0]['BMI'], 19.53)

    def test_wrong_review_version_fails(self):
        with self.assertRaises(ValueError):
            apply_review(self.normalize([source('a', 60, 100)]),
                         [dict(record_id='a', expected_weight='99', operation='exclude', reason='wrong')])

    def test_duplicate_id_and_nonfinite_fail(self):
        for rows in ([source('a', 60, 50), source('a', 80, 51)], [source('a', 60, 'NaN')]):
            with self.assertRaises(ValueError):
                self.normalize(rows)

    def test_missing_weight_is_preserved_and_flagged(self):
        rows = screen(self.normalize([source('a', 60, '')]), CONFIG)
        self.assertIsNone(rows[0]['weight'])
        self.assertIn('missing_weight', rows[0]['flags'])

    def test_cli_outputs_and_no_overwrite(self):
        original = (ROOT / 'examples/synthetic.csv').read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'result'
            summary = run(ROOT / 'examples/synthetic.csv', ROOT / 'examples/config.json', out, ROOT / 'examples/review.csv')
            self.assertEqual(summary['input_rows'], 7)
            self.assertEqual(summary['output_rows'], 7)
            self.assertEqual(summary['reviewed_rows'], 1)
            self.assertEqual(len(list(out.glob('*'))), 4)
            with self.assertRaises(FileExistsError):
                run(ROOT / 'examples/synthetic.csv', ROOT / 'examples/config.json', out)
        self.assertEqual((ROOT / 'examples/synthetic.csv').read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
