"""Export actual distance/similarity values at the subset grid winners."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from ctg.provenance import atomic_write_csv, atomic_write_json


def main(out):
    meta = json.loads((out/'manifest.json').read_text())
    assert meta['status']=='complete'
    assert json.loads((out/'validation.json').read_text())['status']=='passed'
    winners = pd.read_csv(out/'best_grid_points.csv')
    timing = pd.read_csv(out/'preparation/dtw_timings.csv')
    summaries, conditions, mapping = [], [], []
    (out/'selected_predictors').mkdir(exist_ok=True)
    for index, choice in winners.drop_duplicates(['tau','k']).reset_index(drop=True).iterrows():
        name = Path(timing.loc[timing.tau.eq(choice.tau),'training_file'].iloc[0]).name
        d = pd.read_csv(out/'preparation'/name)
        assert not d.condition_id.isin(meta['excluded_conditions']).any()
        assert 'similarity_exp_k' not in d.columns
        d['tau'] = choice.tau
        d['k'] = choice.k
        d['similarity_exp_k'] = np.exp(-choice.k*d.raw_distance)
        a = -choice.k*(d.raw_distance-d.raw_distance.min())
        shifted = np.expm1(a) if np.max(np.abs(a))<.1 else np.exp(a)
        d['similarity_z'] = (shifted-shifted.mean())/shifted.std(ddof=1)
        np.testing.assert_allclose([shifted.mean(),shifted.std(ddof=1)],
                                  [choice.predictor_shifted_mean,choice.predictor_sd],rtol=1e-9,atol=1e-13)
        if d.similarity_exp_k.std()>1e-12:
            np.testing.assert_allclose(d.similarity_z,
                (d.similarity_exp_k-d.similarity_exp_k.mean())/d.similarity_exp_k.std(),atol=1e-8)
        filename = f'candidate_{index:02d}.csv'
        atomic_write_csv(out/'selected_predictors'/filename,d)
        for w in winners.loc[winners.tau.eq(choice.tau)&winners.k.eq(choice.k)].itertuples():
            mapping.append(dict(scope=w.scope, objective=w.objective,tau=w.tau,k=w.k,
                                table=f'selected_predictors/{filename}'))
        for field in ['raw_distance','similarity_exp_k','similarity_z']:
            v=d[field]
            record=dict(tau=choice.tau,k=choice.k,field=field,n=len(v),mean=v.mean(),sd=v.std(),
                        minimum=v.min(),maximum=v.max(),unique_values=v.nunique())
            record.update({f'p{q:02d}':v.quantile(q/100) for q in [1,5,25,50,75,95,99]})
            summaries.append(record)
        for condition,g in d.groupby('condition_id'):
            conditions.append(dict(tau=choice.tau,k=choice.k,condition_id=condition,n=len(g),
                                   distance_mean=g.raw_distance.mean(),distance_min=g.raw_distance.min(),
                                   distance_max=g.raw_distance.max(),similarity_mean=g.similarity_exp_k.mean(),
                                   similarity_sd=g.similarity_exp_k.std(),similarity_min=g.similarity_exp_k.min(),
                                   similarity_max=g.similarity_exp_k.max(),below_001=int(g.similarity_exp_k.lt(.01).sum()),
                                   above_099=int(g.similarity_exp_k.gt(.99).sum()),
                                   zero_distances=int(g.raw_distance.eq(0).sum())))
    atomic_write_csv(out/'selected_predictor_index.csv',pd.DataFrame(mapping))
    atomic_write_csv(out/'selected_predictor_distribution.csv',pd.DataFrame(summaries))
    atomic_write_csv(out/'selected_predictor_by_condition.csv',pd.DataFrame(conditions))
    old=pd.read_csv(Path(meta['source_landscape'])/'best_grid_points.csv')
    cols=['scope','objective','tau','k','z','log_likelihood','n_rows','participants']
    comparison=pd.concat([old[cols].assign(sample='All four conditions'),
                          winners[cols].assign(sample='Talker-specific excluded')],ignore_index=True)
    atomic_write_csv(out/'selected_parameters_comparison.csv',comparison)
    atomic_write_json(out/'selected_predictors_validation.json',dict(status='passed',
        distinct_candidates=len(winners.drop_duplicates(['tau','k'])),
        excluded_condition_rows=0,scaling_reproduces_fits=True,
        descriptive_weighting='one row per retained training word response',
        warning='Raw z and total fitted likelihood are not performance comparisons across different samples'))
    print(comparison.to_string(index=False))
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    main(p.parse_args().input)
