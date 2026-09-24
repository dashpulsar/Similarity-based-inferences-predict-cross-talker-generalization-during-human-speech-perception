from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ctg.ceiling_cv import _prepare_keys, compute_cross_validated_ceiling
from ctg.config import load_project


class ConditionalCeilingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = pd.DataFrame([
            dict(participant_id=f'{condition}-{fold}', phase='test', item_id='AN19.KOR.T.W1',
                 item_talker='T', response_expected='word', response_correct=correct,
                 response_incorrect=1-correct, exposure_test_condition_id=condition,
                 **{'exposure_test_condition.original': condition})
            for condition, correct in [('A', 1), ('B', 0)] for fold in range(3)
        ])
        self.rows.to_csv(self.root/'behavior.csv', index=False)
        pd.DataFrame(dict(participant_id=self.rows.participant_id, fold=[0,1,2]*2)).to_csv(
            self.root/'folds.csv', index=False)
        config = load_project(Path(__file__).resolve().parents[1]/'configs/project.json')
        self.spec = replace(config.dataset('AN19'), behavior=self.root/'behavior.csv')

    def run_ceiling(self, **kwargs):
        return compute_cross_validated_ceiling(spec=self.spec, folds_path=self.root/'folds.csv',
                                               output_dir=self.root/'out', **kwargs)

    def test_conditions_stay_separate(self):
        p, m = self.run_ceiling()
        np.testing.assert_allclose(p.loc[p.ceiling_condition_id.eq('A'), 'predicted_probability'], 2.5/3)
        np.testing.assert_allclose(p.loc[p.ceiling_condition_id.eq('B'), 'predicted_probability'], .5/3)
        self.assertTrue(m.status.eq('complete').all())

    def test_heldout_responses_do_not_change_own_fold_predictions(self):
        before, _ = self.run_ceiling()
        self.rows.loc[self.rows.participant_id.str.endswith('-0'), ['response_correct','response_incorrect']] = [0,1]
        self.rows.to_csv(self.root/'behavior.csv', index=False)
        after, _ = self.run_ceiling()
        np.testing.assert_array_equal(before.loc[before.fold.eq(0), 'predicted_probability'],
                                      after.loc[after.fold.eq(0), 'predicted_probability'])

    def test_x21_repeated_keyword_keeps_sentence_context(self):
        d = pd.concat([self.rows.iloc[:1]]*2, ignore_index=True)
        d['item_id'] = ['X21.CMN.T.HT1_S001.Wthe', 'X21.CMN.T.HT1_S002.Wthe']
        keys = _prepare_keys('X21', d)
        self.assertIn('ceiling_sentence_id', keys)
        self.assertEqual(len(d[keys].drop_duplicates()), 2)

    def test_unseen_cell_rejected_or_explicitly_unavailable(self):
        self.rows.loc[0, 'response_expected'] = 'unique'
        self.rows.to_csv(self.root/'behavior.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'absent from training'):
            self.run_ceiling()
        p, m = self.run_ceiling(missing_cell_policy='mark_unavailable')
        self.assertEqual(p.prediction_status.eq('unseen_training_cell').sum(), 1)
        self.assertTrue(p.loc[p.prediction_status.ne('available'), 'log_loss'].isna().all())
        overall = m.loc[m.scope.eq('oof_all')].iloc[0]
        self.assertEqual((overall.total_trials, overall.requested_total_trials), (5,6))
        self.assertEqual(overall.status, 'partial_coverage')

    def test_grouped_counts_use_word_denominator(self):
        self.rows[['response_correct','response_incorrect']] *= 5
        self.rows.to_csv(self.root/'behavior.csv', index=False)
        p, m = self.run_ceiling()
        a = m.loc[m.scope.eq('oof_all')].iloc[0]
        self.assertEqual(a.total_trials, 30)
        self.assertAlmostEqual(a.mean_log_loss, p.log_loss.sum()/30)

    def test_duplicate_participant_folds_rejected(self):
        f = pd.read_csv(self.root/'folds.csv')
        pd.concat([f,f.iloc[:1]]).to_csv(self.root/'folds.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.run_ceiling()


if __name__ == '__main__':
    unittest.main()
