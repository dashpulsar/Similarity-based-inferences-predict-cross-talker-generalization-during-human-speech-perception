"""Bounded training-only feasibility test of joint tau/k optimization.

Uses AN19 base t-SNE Tr-24, outer fold 0 held out (never scored here).
Does not overwrite production results. Time limits stop this script's own fits.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'src'))
from ctg.config import load_project
from ctg.features import FeatureStore
from ctg.metrics import dtw_distance
from ctg.provenance import atomic_write_csv, atomic_write_json, sha256_file, runtime_record


class TimeBudgetExceeded(RuntimeError):
    pass


class Probe:
    def __init__(self, out, rscript, budget, workers):
        self.out, self.rscript = out, rscript
        if out.exists():
            raise FileExistsError(f'Use a fresh output directory: {out}')
        out.mkdir(parents=True)
        self.start = time.perf_counter()
        self.deadline = self.start + budget
        self.workers = workers
        self.records, self.timing, self.traces = [], [], []
        self.cache, self.distance_cache = {}, {}
        self.calls = 0
        derived = PROJECT/'artifacts/derived'
        self.input_path = derived/'AN19-AN19_hubert_base_tsne-confirmatory-tr24-20260906-model-input.csv'
        data = pd.read_csv(self.input_path)
        self.train = data.loc[data.fold.ne(0) & data.predictor_status.eq('available')].copy().reset_index(drop=True)
        assert self.train.feature_key.eq('tr_24').all()
        self.keys = ['condition_id','behavior_item_id','response_expected','exposure_set_id']
        cells_path = derived/'AN19-pairs/cells.csv'
        pairs_path = derived/'AN19-pairs/pairs.csv'
        cells = pd.read_csv(cells_path)
        wanted = self.train[self.keys].drop_duplicates()
        cells = cells.merge(wanted, on=self.keys, validate='many_to_one')
        assert cells.cell_status.eq('available').all()
        assert cells.groupby('cell_id').source_talker_id.nunique().eq(cells.groupby('cell_id').n_expected_source_talkers.first()).all()
        # AN19 units are whole words; reject interval mappings in this focused probe.
        pairs = pd.read_csv(pairs_path)
        self.pairs = pairs.loc[pairs.pair_id.isin(cells.pair_id)].copy().reset_index(drop=True)
        assert self.pairs.pair_id.nunique()==cells.pair_id.nunique()
        assert self.pairs[['test_start_seconds','source_start_seconds']].isna().all().all()
        lookup = dict(zip(self.pairs.pair_id, range(len(self.pairs))))
        cells['pair_index'] = cells.pair_id.map(lookup)
        self.cells = cells
        self.cell_metadata = cells[['cell_id']+self.keys].drop_duplicates()
        assert not self.cell_metadata.duplicated(self.keys).any()
        spec = load_project(PROJECT/'configs/project.json').store('AN19_hubert_base_tsne')
        t0 = time.perf_counter()
        units = set(zip(self.pairs.test_speaker_id,self.pairs.test_unit_id)) | set(zip(self.pairs.source_speaker_id,self.pairs.source_unit_id))
        with FeatureStore(spec) as store:
            features = {key: np.ascontiguousarray(store.read(*key,'tr_24')) for key in sorted(units)}
        self.sequences = [(features[(r.test_speaker_id,r.test_unit_id)], features[(r.source_speaker_id,r.source_unit_id)]) for r in self.pairs.itertuples()]
        # Include compilation/startup time separately, outside per-tau timings.
        dtw_distance(*self.sequences[0], tau=2.)
        self.setup_seconds = time.perf_counter()-t0
        self.manifest = dict(**runtime_record(), scope='Feasibility only: AN19 base 3-D t-SNE Tr-24; folds 1+2 training; fold 0 excluded',
            responses=len(self.train), participants=self.train.participant_id.nunique(), physical_pairs=len(self.pairs),
            normalization='mean_sequence_length', coordinate_scaling='none',
            aggregation='mean physical distances within source talker, then equal mean over source talkers, then exp(-k*d)',
            model='binomial-logit: similarity_z + (1|participant_id) + (1|analysis_item_id) + (1|test_talker_id)',
            glmm_optimizer='bobyqa', glmm_maxfun=20000, likelihood='lme4 logLik, Laplace nAGQ=1',
            objectives=['signed training Wald z','training fitted GLMM log likelihood'], penalty='none',
            structure_fallback=False, outer_test_used=False, hve_run=False,
            training_scaling='Each candidate similarity standardized over training responses only',
            tau_bounds=[1.,3.], k_bounds=[.001,2.], search_coordinates='tau, log10(k)',
            scope_of_bounds='Preliminary timing/search box, not final scientific parameter bounds',
            seconds_budget=budget, workers=workers, feature_load_and_warmup_seconds=self.setup_seconds,
            sources={str(p):sha256_file(p) for p in [self.input_path,cells_path,pairs_path,Path(__file__),PROJECT/'R/fit_sbi_optimizer_probe.R',PROJECT/'src/ctg/metrics.py']},
            feature_store=str(spec.path),feature_store_size=spec.path.stat().st_size, feature_store_mtime_ns=spec.path.stat().st_mtime_ns)
        self.save()

    def check_time(self):
        if time.perf_counter() >= self.deadline:
            raise TimeBudgetExceeded('Wall-time budget reached')

    def save(self):
        atomic_write_json(self.out/'manifest.json',self.manifest)
        if self.records: atomic_write_csv(self.out/'evaluations.csv',pd.DataFrame(self.records))
        if self.timing: atomic_write_csv(self.out/'dtw_timings.csv',pd.DataFrame(self.timing))
        if self.traces: atomic_write_csv(self.out/'search_trace.csv',pd.DataFrame(self.traces))

    def distances(self,tau):
        token=float(tau).hex()
        if token in self.distance_cache: return self.distance_cache[token]
        self.check_time()
        t0=time.perf_counter()
        def calculate(chunk):
            values=[]
            for a,b in chunk:
                self.check_time()
                values.append(dtw_distance(a,b,tau=tau).distance)
            return values
        chunks=[self.sequences[i:i+128] for i in range(0,len(self.sequences),128)]
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            values=np.concatenate(list(executor.map(calculate,chunks)))
        d=self.cells[['cell_id','source_talker_id','pair_index']].copy()
        d['distance']=values[d.pair_index.to_numpy()]
        by_source=d.groupby(['cell_id','source_talker_id'],sort=True).distance.mean()
        by_cell=by_source.groupby('cell_id').mean().rename('candidate_distance').reset_index()
        mapped=self.cell_metadata.merge(by_cell,on='cell_id',validate='one_to_one')
        frame=self.train.merge(mapped[self.keys+['candidate_distance']],on=self.keys,validate='many_to_one',sort=False)
        pd.testing.assert_frame_equal(frame[self.keys],self.train[self.keys])
        assert len(frame)==len(self.train) and np.isfinite(frame.candidate_distance).all()
        if tau==2.:
            error=float(np.max(np.abs(frame.raw_distance-frame.candidate_distance)))
            self.manifest['tau2_historical_distance_max_error']=error
            np.testing.assert_allclose(frame.raw_distance,frame.candidate_distance,rtol=1e-10,atol=1e-10)
        frame['raw_distance']=frame.pop('candidate_distance')
        number=len(self.timing)
        path=self.out/f'tau_{number:04d}_training.csv'
        atomic_write_csv(path,frame)
        dt=time.perf_counter()-t0
        self.timing.append(dict(tau=tau,seconds=dt,physical_pairs=len(values),training_file=path.name))
        self.distance_cache[token]=(path,dt)
        self.save()
        return path,dt

    def evaluate(self,tau,k):
        self.check_time()
        if not (1<=tau<=3 and .001-1e-12<=k<=2+1e-12): raise ValueError('Outside pilot bounds')
        self.calls+=1
        key=(float(tau).hex(),float(k).hex())
        if key in self.cache: return self.cache[key],True
        t0=time.perf_counter()
        cached_tau=float(tau).hex() in self.distance_cache
        path,dt=self.distances(tau)
        self.check_time()
        number=len(self.records)
        output=self.out/f'fit_{number:04d}.csv'
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        status=None
        try:
            completed=subprocess.run([str(self.rscript),str(PROJECT/'R/fit_sbi_optimizer_probe.R'),str(path),str(output),str(k)],
                capture_output=True,text=True,env=env,timeout=max(.1,min(90.,self.deadline-time.perf_counter())))
            (self.out/f'fit_{number:04d}.log').write_text(completed.stdout+'\n'+completed.stderr,encoding='utf-8')
            result=pd.read_csv(output).iloc[0].to_dict() if completed.returncode==0 and output.exists() else {'status':'process_failed'}
        except subprocess.TimeoutExpired:
            result={'status':'timeout'}
        result.update(evaluation=number,tau=float(tau),k=float(k),wall_seconds=time.perf_counter()-t0,
            dtw_cached=cached_tau,dtw_seconds=0. if cached_tau else dt)
        self.records.append(result)
        self.cache[key]=result
        self.save()
        print(json.dumps({key:result.get(key) for key in ['evaluation','tau','k','status','z','log_likelihood','wall_seconds']},default=str),flush=True)
        return result,False

    def benchmark(self,trials):
        import optuna
        from scipy.optimize import differential_evolution
        import scipy
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.manifest.update(optuna_version=optuna.__version__,scipy_version=scipy.__version__,
            benchmark_trials_per_method_objective=trials,benchmark_seeds=[42],benchmark_status='running')
        self.save()
        for objective in ['z','log_likelihood']:
            for method in ['grid','tpe','differential_evolution']:
                counter=0
                best=-np.inf
                started=time.perf_counter()
                def score(x):
                    nonlocal counter,best
                    if counter>=trials: raise TimeBudgetExceeded('Candidate quota reached')
                    record,cached=self.evaluate(float(x[0]),float(10.**x[1]))
                    valid=record['status']=='ok' and np.isfinite(record.get(objective,np.nan))
                    value=float(record[objective]) if valid else -np.inf
                    best=max(best,value)
                    counter+=1
                    self.traces.append(dict(method=method,objective=objective,seed=42,trial=counter,
                        evaluation=record['evaluation'],tau=record['tau'],k=record['k'],score=value if valid else np.nan,
                        best_score=best if np.isfinite(best) else np.nan,valid=valid,
                        full_evaluation_cached=cached,elapsed_seconds=time.perf_counter()-started,
                        evaluation_wall_seconds=record['wall_seconds']))
                    self.save()
                    return -value if valid else 1e12
                bounds=[(1.,3.),(np.log10(.001),np.log10(2.))]
                if method=='grid':
                    for tau,logk in product(np.linspace(1,3,3),np.linspace(*bounds[1],4)):
                        if counter>=trials: break
                        score([tau,logk])
                elif method=='tpe':
                    study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=5))
                    study.optimize(lambda t:score([t.suggest_float('tau',1.,3.),t.suggest_float('log10_k',*bounds[1])]),n_trials=trials)
                else:
                    # Six population members and one generation: exactly twelve evaluations.
                    differential_evolution(score,bounds,seed=42,popsize=3,maxiter=1,polish=False,tol=0,atol=0,workers=1)
        self.manifest['benchmark_status']='complete_tiny_one_seed_check'
        self.save()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rscript',type=Path,required=True)
    parser.add_argument('--budget-seconds',type=float,default=600)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--benchmark',action='store_true')
    args=parser.parse_args()
    probe=Probe(args.output,args.rscript,args.budget_seconds,args.workers)
    status='probe_complete'
    try:
        for tau,k in product([2.,1.,3.],[.001,.05,2.]): probe.evaluate(tau,k)
        records=pd.DataFrame(probe.records)
        dt=np.median([r['seconds'] for r in probe.timing])
        fit=np.median(records.wall_seconds-records.dtw_seconds)
        # Full illustration: 3 methods x 2 objectives x 3 seeds x 40 calls.
        estimate=720*(dt+fit)
        probe.manifest.update(estimated_720_call_serial_seconds=estimate,median_fit_including_R_startup_seconds=fit,
            median_dtw_seconds=dt,estimate_note='One layer, one fold; conservative new tau on each call; no repeated-tau savings or outer parallelism assumed')
        # Tiny comparison adds at most 72 proposals, shared exact computation cache.
        remaining=probe.deadline-time.perf_counter()
        if args.benchmark and 72*(dt+fit)<remaining and records.status.eq('ok').all():
            probe.benchmark(12)
            status='tiny_benchmark_complete'
        elif args.benchmark:
            status='paused_after_probe_cost_or_fit_gate'
    except TimeBudgetExceeded as e:
        status='paused_time_budget'
        probe.manifest['pause_reason']=str(e)
    finally:
        probe.manifest.update(status=status,total_elapsed_seconds=time.perf_counter()-probe.start)
        probe.save()
        print(json.dumps(dict(status=status,elapsed_seconds=probe.manifest['total_elapsed_seconds']),default=str),flush=True)


if __name__=='__main__': main()
