"""Recompute conditional behavioral reference predictions; preserve old results.

This checks likelihood references only. It does not create a compatible Wald-z
ceiling or normalize SBI/HVE fits that use a different response sample.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'src'))
from ctg.config import load_project
from ctg.ceiling_cv import compute_cross_validated_ceiling, _prepare_keys
from ctg.provenance import atomic_write_csv, atomic_write_json, runtime_record, sha256_file


def main():
    config = load_project(PROJECT/'configs/project.json')
    out = PROJECT/'analysis/model_comparison/conditional_ceiling'
    comparisons, coverage, sources, context = [], [], [], []
    for dataset in ['AN19','X21','B23']:
        folds = PROJECT/f'artifacts/derived/{dataset}-folds.csv'
        historical = PROJECT/f'analysis/model_comparison/reference_checks/ceilings/{dataset}/oof_predictions.csv'
        old = pd.read_csv(historical)
        new, metrics = compute_cross_validated_ceiling(
            spec=config.dataset(dataset), folds_path=folds, output_dir=out/dataset,
            missing_cell_policy='mark_unavailable')
        keys = ['participant_id','fold','item_id','response_expected']
        joined = new.merge(old[keys+['predicted_probability','log_loss','response_correct','response_incorrect']],
                           on=keys, how='outer', suffixes=('_new','_old'), validate='one_to_one', indicator=True)
        if not joined['_merge'].eq('both').all():
            raise ValueError('Historical/current response identities differ')
        for field in ['response_correct','response_incorrect']:
            np.testing.assert_array_equal(joined[field+'_new'], joined[field+'_old'])
        atomic_write_csv(out/dataset/'paired_old_new_predictions.csv', joined.drop(columns='_merge'))
        for fold in [0,1,2,None]:
            d = joined if fold is None else joined.loc[joined.fold.eq(fold)]
            common = d.loc[d.prediction_status.eq('available')]
            n = common.n_trials.sum()
            comparisons.append(dict(dataset=dataset, fold=fold, scope='all' if fold is None else 'fold',
                requested_rows=len(d), paired_rows=len(common), paired_word_responses=int(n),
                missing_rows=len(d)-len(common),
                old_mean_loss_on_common=float(common.log_loss_old.sum()/n) if n else np.nan,
                new_mean_loss_on_common=float(common.log_loss_new.sum()/n) if n else np.nan,
                new_minus_old=float((common.log_loss_new-common.log_loss_old).sum()/n) if n else np.nan,
                max_probability_difference=float((common.predicted_probability_new-common.predicted_probability_old).abs().max())))
        for (fold, condition), d in new.groupby(['fold','exposure_test_condition_id']):
            coverage.append(dict(dataset=dataset, fold=int(fold), condition=condition,
                requested_rows=len(d), available_rows=int(d.prediction_status.eq('available').sum()),
                unavailable_rows=int(d.prediction_status.ne('available').sum())))
        # Quantify the additional X21 context distinction independently of condition.
        behavior = pd.read_csv(config.dataset(dataset).behavior)
        rows = behavior.loc[behavior.phase.eq('test')].copy()
        _prepare_keys(dataset, rows)
        if dataset == 'X21':
            c = rows.groupby(['ceiling_condition_id','ceiling_talker_id','ceiling_content_id']).ceiling_sentence_id.nunique()
            context.append(dict(dataset=dataset, old_keyword_condition_cells=len(c),
                                cells_spanning_multiple_sentences=int(c.gt(1).sum()), max_sentences=int(c.max())))
        sources.append(dict(dataset=dataset, historical_predictions=str(historical),
                            historical_sha256=sha256_file(historical), folds_sha256=sha256_file(folds),
                            behavior_sha256=sha256_file(config.dataset(dataset).behavior)))
        print(dataset, metrics[['scope','fold','missing_rows','mean_log_loss']].to_dict('records'), flush=True)
    summary = pd.DataFrame(comparisons)
    atomic_write_csv(out/'comparison.csv', summary)
    atomic_write_csv(out/'coverage_by_condition_fold.csv', pd.DataFrame(coverage))
    atomic_write_csv(out/'keyword_context_audit.csv', pd.DataFrame(context))
    fig, axes = plt.subplots(1,3, figsize=(12,4), constrained_layout=True)
    for ax, dataset in zip(axes, ['AN19','X21','B23']):
        d = summary.loc[summary.dataset.eq(dataset) & summary.scope.eq('fold')]
        for _, row in d.iterrows():
            ax.plot([0,1], [row.old_mean_loss_on_common, row.new_mean_loss_on_common],
                    marker='o', color='0.45', alpha=.7)
        ax.set_xticks([0,1], ['Previous','Condition/context'])
        ax.set_title(dataset)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Mean held-out log loss\n(on identical available responses)')
    for ext in ['png','pdf','svg']:
        fig.savefig(out/f'conditional_ceiling_comparison.{ext}', dpi=180)
    plt.close(fig)
    atomic_write_json(out/'audit_record.json', dict(**runtime_record(), sources=sources,
        script_sha256=sha256_file(Path(__file__)), scope='All behavioral test rows; paired comparisons on common available rows',
        nested_cv_changed=False, legacy_z_ceiling_changed=False,
        note='Partial-coverage references cannot normalize complete-sample predictors without explicit row matching. No pooled-condition fallback.'))


if __name__ == '__main__':
    main()
