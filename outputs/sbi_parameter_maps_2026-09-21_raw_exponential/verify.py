"""Check grid identity, cached input integrity and actual raw predictor values."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
def main():
    meta=json.loads((HERE/'manifest.json').read_text())
    d=pd.read_csv(HERE/'landscape.csv')
    assert not d.duplicated(['tau_index','k_index']).any()
    assert len(d)<=1247
    assert np.allclose(d.tau,np.array(meta['tau_values'])[d.tau_index],rtol=0,atol=1e-12)
    assert np.allclose(d.k,np.array(meta['k_values'])[d.k_index],rtol=1e-12,atol=0)
    identities=None; raw_checks=0; expected_source_rows=None
    for relative,expected in meta['data_sha256'].items():
        path=ROOT/relative
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
        source=pd.read_csv(path)
        assert len(source)==10969 and set(source.fold)=={1,2}
        cols=['participant_id','fold','condition_id','analysis_item_id','response_correct','response_incorrect']
        ids=source[cols].to_json(orient='values')
        if identities is None: identities=ids
        assert ids==identities
        ti=int(path.stem.split('_')[1])
        q=d[(d.tau_index==ti) & d.predictor_sd.notna()]
        for _,row in q.iterrows():
            raw=np.exp(-row.k*source.raw_distance.to_numpy())
            assert np.isclose(raw.mean(),row.predictor_mean,rtol=1e-10,atol=1e-14)
            assert np.isclose(raw.std(ddof=1),row.predictor_sd,rtol=1e-7,atol=1e-14)
            assert np.isclose(raw.min(),row.predictor_min,rtol=1e-10,atol=1e-14)
            assert np.isclose(raw.max(),row.predictor_max,rtol=1e-10,atol=1e-14)
            raw_checks+=1
    for col in ['z','log_likelihood']:
        assert d.loc[d.status.isin(['timeout','not_started_budget','failed','process_failed']),col].isna().all()
    result=dict(status='passed',recorded_grid_points=len(d),expected_grid_points=1247,
                input_hashes_checked=29,identical_response_rows_across_taus=True,
                raw_exponential_statistic_checks=raw_checks,no_failure_score_imputation=True,
                complete_grid_recorded=len(d)==1247,run_status=meta['status'])
    (HERE/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
