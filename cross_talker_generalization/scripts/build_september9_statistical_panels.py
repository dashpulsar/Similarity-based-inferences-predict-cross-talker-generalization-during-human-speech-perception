"""Redraw retained fold statistics for the annotated-report response; no model fits."""
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from build_z_value_review import LAYERS, DATASETS, fold_ci, references, plot_points, label, measure_label

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO / 'cross_talker_generalization'
OUT = PROJECT / 'analysis/speech'
SOURCES = {}
INVENTORY = []


def read(path, scope='current_repository'):
    frame = pd.read_csv(path)
    SOURCES[f'{scope}/{path.name}/{path.parent.name}'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return frame


def save(fig, name, scope):
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / 'figures' / f'{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    INVENTORY.append(dict(figure=name, scope=scope))


def z_figures():
    path = PROJECT / 'analysis/model_comparison/reference_checks/z_value_review/tables'
    frame = read(path / 'historical_sbi_fold_z.csv')
    ceilings = read(path / 'historical_ceiling_fold_z.csv')
    other = read(path / 'historical_hve_ceiling_fold_z.csv')
    assert np.allclose(ceilings.z_value, other.z_value)
    keys = ['mfcc39', 'strf24_legacy', *LAYERS]
    positions = [0, 1, *range(3, len(LAYERS)+3)]
    handles = [Line2D([], [], color='#b56e00', linestyle=':', label='Significance threshold (z = +/- 1.96)'),
               Patch(facecolor='#bbbbbb', alpha=.4, label='Ceiling'),
               Line2D([], [], marker='o', color='#777777', linestyle='none', markersize=4, label='Individual folds')]
    for dataset in DATASETS:
        ceiling = ceilings.loc[ceilings.dataset_id.eq(dataset)].sort_values('fold').z_value.to_numpy()
        for variant in ('base', 'ft'):
            group = frame.loc[frame.dataset_id.eq(dataset) & frame.source.isin([variant, 'mfcc39', 'strf24_legacy'])]
            fig, axes = plt.subplots(2, 1, figsize=(10, 7.4), layout='constrained')
            for ax, percent in zip(axes, (True, False)):
                references(ax, ceiling, percent)
                plot_points(ax, group, keys[:2], positions[:2], 'black', percent, ceiling, connect=False)
                plot_points(ax, group, keys[2:], positions[2:], 'black', percent, ceiling)
                ax.set_xticks(positions, [label(k) for k in keys], rotation=50, ha='right', fontsize=8)
                ax.set_xlabel('Feature space')
                ax.margins(x=.02, y=.1)
            # All stored maxima are <69%; ceiling lower intervals are >88%.
            # A single legend row at 80% avoids both curves and the ceiling band.
            transform = axes[0].get_yaxis_transform()
            axes[0].legend(handles=handles, loc='center', bbox_to_anchor=(.5, 80),
                           bbox_transform=transform, ncol=3, fontsize=7, frameon=False)
            fig.suptitle(f"{dataset} | SBI | HuBERT {'base' if variant=='base' else 'ASR-FT'}", fontsize=13)
            save(fig, f'sbi_{dataset.lower()}_{variant}_z', 'retained notebook test-fold refit z; shared historical behavioral ceiling')
    frame.to_csv(OUT / 'tables/sbi_retained_fold_z.csv', index=False)
    ceilings.to_csv(OUT / 'tables/shared_historical_ceiling_z.csv', index=False)


def likelihood_figures():
    # Recover compact metrics from the linked original worktree; never reset/checkout.
    git_common = Path(subprocess.check_output(['git','rev-parse','--path-format=absolute','--git-common-dir'], cwd=REPO, text=True).strip())
    old = git_common.parent / 'cross_talker_generalization/artifacts/models'
    all_rows = []
    for dataset in DATASETS:
        baseline = read(old / f'{dataset}-{dataset}_acoustic-confirmatory/cv_metrics.csv', 'retained_worktree')
        baseline = baseline.loc[baseline.scope.eq('oof_fold') & baseline.model_id.eq('M_predictor')].copy()
        for variant in ('base','ft'):
            neural = read(old / f'{dataset}-{dataset}_hubert_{variant}_tsne-confirmatory/cv_metrics.csv', 'retained_worktree')
            neural = neural.loc[neural.scope.eq('oof_fold') & neural.model_id.eq('M_predictor')].copy()
            assert set(neural.feature_key) == set(LAYERS)
            group = pd.concat([baseline, neural], ignore_index=True)
            assert group.groupby('feature_key').size().eq(3).all()
            assert group.groupby('fold').total_trials.nunique().eq(1).all()
            group['layer'] = group.feature_key
            group['z_value'] = -group.mean_log_loss  # generic fold plotting helper; never exported as z
            group['held_out_log_likelihood_per_trial'] = -group.total_log_loss/group.total_trials
            assert np.allclose(group.z_value, group.held_out_log_likelihood_per_trial)
            group['variant'] = variant
            group['predictor_definition'] = 'negative raw DTW, standardized within training folds; retained confirmatory run'
            all_rows.append(group.drop(columns=['z_value','layer']))
            keys = ['mfcc39','strf24_legacy',*LAYERS]
            pos = [0,1,*range(3,len(LAYERS)+3)]
            fig, ax = plt.subplots(figsize=(10,4.6), layout='constrained')
            plot_points(ax, group, keys[:2], pos[:2], 'black', False, None, connect=False)
            plot_points(ax, group, keys[2:], pos[2:], 'black', False, None)
            ax.set_xticks(pos,[label(k) for k in keys],rotation=50,ha='right',fontsize=8)
            ax.set(xlabel='Feature space',ylabel='Held-out log likelihood per word',
                   title=f"{dataset} | SBI | HuBERT {'base' if variant=='base' else 'ASR-FT'}")
            ax.spines[['top','right']].set_visible(False)
            fig.text(.5,-.025,'Predictor-only model | Higher is better | Mean and 95% three-fold bootstrap interval',ha='center',fontsize=8)
            save(fig,f'sbi_{dataset.lower()}_{variant}_loglik','retained predictor-only out-of-fold scores; no compatible item-by-condition reference inserted')
    pd.concat(all_rows,ignore_index=True).to_csv(OUT / 'tables/sbi_retained_fold_loglik.csv',index=False)
    hve_rows = []
    for dataset in DATASETS:
        parts=[]
        for variant in ('base','ft'):
            suffix='selection' if dataset=='B23' else 'revised-selection'
            part=read(PROJECT / f'artifacts/models/{dataset}-HVE-{variant}-{suffix}/cv_metrics.csv')
            part=part.loc[part.scope.eq('oof_fold') & part.model_id.eq('M_predictor')].copy()
            part[['layer','measure']]=part.feature_key.str.split('::',expand=True)
            part['variant']=variant
            part['held_out_log_likelihood_per_trial']=-part.total_log_loss/part.total_trials
            part['sample_stratum']=np.where((dataset=='B23') & part.measure.eq('overall_order_sensitive'),'97 ordered participants','168 participants' if dataset=='B23' else 'all test participants, including control' if dataset=='X21' else 'all exposure participants')
            parts.append(part)
        frame=pd.concat(parts,ignore_index=True)
        hve_rows.append(frame)
        measures=sorted(frame.measure.unique())
        for start in range(0,len(measures),4):
            fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
            for ax,method in zip(axes.flat,measures[start:start+4]):
                for variant,color,offset in [('base','#20639b',-.1),('ft','#b43b35',.1)]:
                    part=frame.loc[frame.measure.eq(method) & frame.variant.eq(variant)].copy()
                    part['z_value']=part.held_out_log_likelihood_per_trial
                    plot_points(ax,part,LAYERS,np.arange(len(LAYERS)),color,False,None,offset=offset)
                ax.set_xticks(range(len(LAYERS)),[label(k) for k in LAYERS],rotation=60,ha='right',fontsize=7)
                ax.set(xlabel='Feature space',ylabel='Held-out log likelihood per word',title=measure_label(method))
                if dataset=='B23':
                    ax.set_title(measure_label(method)+'\n'+('97 ordered participants' if method=='overall_order_sensitive' else '168 participants'),fontsize=10)
                ax.spines[['top','right']].set_visible(False)
            for ax in list(axes.flat)[len(measures[start:start+4]):]: ax.set_visible(False)
            fig.suptitle(f'{dataset} | HVE | Predictor-only held-out log likelihood',fontsize=13)
            fig.legend(handles=[Line2D([],[],color=c,label=l) for c,l in [('#20639b','HuBERT base'),('#b43b35','HuBERT ASR-FT')]],loc='outside lower center',ncol=2,frameon=False)
            save(fig,f'hve_{dataset.lower()}_loglik_{start//4+1:02d}','retained revised HVE; three folds; separate B23 samples; not cross-sample ranking')
    pd.concat(hve_rows,ignore_index=True).to_csv(OUT / 'tables/hve_retained_fold_loglik.csv',index=False)


def main():
    (OUT / 'figures').mkdir(parents=True,exist_ok=True)
    (OUT / 'tables').mkdir(exist_ok=True)
    z_figures()
    likelihood_figures()
    pd.DataFrame(INVENTORY).to_csv(OUT / 'tables/statistical_figure_inventory.csv',index=False)
    (OUT / 'tables/statistical_panel_provenance.json').write_text(json.dumps({'source_sha256':SOURCES,'new_GLMM_fits':0,'bootstrap':'all 27 ordered resamples of 3 fold means','note':'Log likelihood omits binomial coefficient constant; per-word Bernoulli score. Do not compare different participant samples.'},indent=2),encoding='utf-8')
    print(f'Saved {len(INVENTORY)} statistical figures',flush=True)


if __name__=='__main__':
    main()
